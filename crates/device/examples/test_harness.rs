// crates/device/examples/test_harness.rs
//
// Minimal test harness for local Phase 1/2 development.
//
// Implements a simple loopback transport over a Unix socket:
// the harness listens for raw request buffers, calls NvidiaBackend::dispatch(),
// and writes the response back.
//
// This is NOT a real vhost-user implementation — it is a straight-line
// request/response loop so we can iterate on the backend without a VM.
// To use it:
//
//   Terminal 1:  cargo run --bin test-harness [--mock] [--socket /tmp/nv.sock]
//   Terminal 2:  socat - UNIX-CONNECT:/tmp/nv.sock   (or use the test client)
//
// With --mock, the harness opens /dev/null instead of /dev/nvidiactl, so
// no NVIDIA GPU is required.  Ioctls will fail with EBADF but open/close
// round-trips work.

use std::io::{Read, Write};
use std::os::unix::net::UnixListener;
use std::path::PathBuf;

use device::nvidia::NvidiaBackend;

// Wire framing: each message is prefixed with a 4-byte little-endian length.
// The harness reads exactly that many bytes as the request, dispatches, then
// writes a 4-byte length prefix + the response.

fn main() {
    env_logger::init();

    let mut socket_path = PathBuf::from("/tmp/nv-vhost.sock");
    let mut mock = false;
    let mut shm_size: u64 = 256 * 1024 * 1024;

    // Minimal arg parser.
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mock" => mock = true,
            "--socket" => {
                i += 1;
                socket_path = PathBuf::from(&args[i]);
            }
            "--shm-size" => {
                i += 1;
                shm_size = args[i].parse().expect("--shm-size must be a number");
            }
            other => eprintln!("unknown arg: {}", other),
        }
        i += 1;
    }

    if mock {
        eprintln!("[test-harness] mock mode: GPU ioctls will fail with EBADF");
    }

    // Remove stale socket file.
    let _ = std::fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path)
        .unwrap_or_else(|e| panic!("bind {:?}: {}", socket_path, e));

    eprintln!("[test-harness] listening on {:?}", socket_path);

    let mut backend = NvidiaBackend::new(shm_size);

    for stream in listener.incoming() {
        match stream {
            Ok(mut conn) => {
                eprintln!("[test-harness] client connected");
                loop {
                    // Read 4-byte length prefix.
                    let mut len_buf = [0u8; 4];
                    match conn.read_exact(&mut len_buf) {
                        Ok(()) => {}
                        Err(_) => break,
                    }
                    let req_len = u32::from_le_bytes(len_buf) as usize;

                    // Read request body.
                    let mut req = vec![0u8; req_len];
                    if conn.read_exact(&mut req).is_err() {
                        break;
                    }

                    // Dispatch.
                    let mut resp = vec![0u8; 8192];
                    let resp_len = backend.dispatch(&req, &mut resp);

                    // Write length-prefixed response.
                    let prefix = (resp_len as u32).to_le_bytes();
                    if conn.write_all(&prefix).is_err() {
                        break;
                    }
                    if conn.write_all(&resp[..resp_len]).is_err() {
                        break;
                    }
                }
                eprintln!("[test-harness] client disconnected");
            }
            Err(e) => eprintln!("[test-harness] accept error: {}", e),
        }
    }
}
