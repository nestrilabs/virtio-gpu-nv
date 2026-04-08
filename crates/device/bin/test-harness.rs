use clap::Parser;
use device::nvidia::NvidiaBackend;
use device::shm::ZoneConfig;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::sync::Mutex;

// Wire framing: each message is prefixed with a 4-byte little-endian length.
// The harness reads exactly that many bytes as the request, dispatches, then
// writes a 4-byte length prefix + the response.

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// Mock mode, opens /dev/null instead of /dev/nvidiactl
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// Socket path to listen on
    #[arg(long, default_value = "/tmp/nv-vhost.sock")]
    socket_path: String,

    /// Total SHM BAR size in bytes (split: 4 MiB UC, half WC, rest WB)
    #[arg(long, default_value_t = 256 * 1024 * 1024)]
    shm_size: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing_subscriber::filter::LevelFilter::INFO.into())
                .from_env()?,
        )
        .init();

    let args = Args::parse();

    if args.mock {
        tracing::warn!("mock mode: GPU ioctls will fail with EBADF");
    }

    // Remove stale socket file.
    let _ = std::fs::remove_file(&args.socket_path);
    let listener = UnixListener::bind(&args.socket_path)
        .unwrap_or_else(|e| panic!("bind {:?}: {}", args.socket_path, e));

    tracing::info!("listening on {:?}", args.socket_path);

    // Build ZoneConfig from the flat shm_size:
    //   4 MiB  → UC  (control registers, doorbells)
    //   half   → WC  (VRAM / framebuffer)
    //   rest   → WB  (DMA buffers)
    let uc = 4 * 1024 * 1024u64;
    let remaining = args.shm_size.saturating_sub(uc);
    let wc = remaining / 2;
    let wb = remaining - wc;
    let cfg = ZoneConfig {
        uc_size: uc,
        wc_size: wc,
        wb_size: wb,
    };

    let backend = Arc::new(Mutex::new(NvidiaBackend::new(cfg)));

    loop {
        tokio::select! {
            // Accept new connections asynchronously.
            accept_res = listener.accept() => {
                match accept_res {
                    Ok((mut conn, _addr)) => {
                        tracing::info!("client connected");
                        let backend_clone = backend.clone();

                        // Spawn a new task per connection to prevent blocking the accept loop.
                        tokio::spawn(async move {
                            loop {
                                // Read 4-byte length prefix.
                                let mut len_buf = [0u8; 4];
                                if conn.read_exact(&mut len_buf).await.is_err() {
                                    break;
                                }
                                let req_len = u32::from_le_bytes(len_buf) as usize;

                                // Read request body.
                                let mut req = vec![0u8; req_len];
                                if conn.read_exact(&mut req).await.is_err() {
                                    break;
                                }

                                // Dispatch.
                                let mut resp = vec![0u8; 8192];
                                let resp_len = {
                                    let mut b = backend_clone.lock().await;
                                    b.dispatch(&req, &mut resp)
                                };

                                // Write length-prefixed response.
                                let prefix = (resp_len as u32).to_le_bytes();
                                if conn.write_all(&prefix).await.is_err() {
                                    break;
                                }
                                if conn.write_all(&resp[..resp_len]).await.is_err() {
                                    break;
                                }
                            }

                            // Close all host fds associated with this client's handles.
                            // This covers the ungraceful disconnect case — the guest
                            // process died without sending NV_MSG_CLOSE for each fd.
                            backend_clone.lock().await.teardown();
                            tracing::info!("client disconnected");
                        });
                    }
                    Err(e) => tracing::error!("accept error: {}", e),
                }
            }

            // Listen for Ctrl-C signal.
            _ = tokio::signal::ctrl_c() => {
                break;
            }
        }
    }

    tracing::info!("exiting cleanly");

    // Final teardown: close any fds that slipped through (e.g. in-flight
    // connections that hadn't disconnected yet when Ctrl-C arrived).
    backend.lock().await.teardown();

    // Remove socket file so a re-run doesn't need --force.
    let _ = std::fs::remove_file(&args.socket_path);

    Ok(())
}
