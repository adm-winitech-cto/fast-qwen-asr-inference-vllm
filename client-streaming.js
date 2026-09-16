/* Client for streaming ASR - this is for TESTING PURPOSES */
import WebSocket from "ws";
import fs from "fs";

const CHUNK_BYTES = 640;
const PCM_BYTES_PER_SECOND = 32000.0;

function parseArgs(argv) {
  let endpoint = "";
  let filePath = "";

  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "-e" && argv[i + 1]) {
      endpoint = argv[i + 1];
      i += 1;
    } else if (argv[i] === "-f" && argv[i + 1]) {
      filePath = argv[i + 1];
      i += 1;
    }
  }

  return { endpoint, filePath };
}

function printUsageAndExit() {
  console.error("Usage: node client-streaming.js -e <ws_url> -f <audio_file>");
  process.exit(1);
}

function streamAudio(ws, filePath) {
  try {
    const fd = fs.openSync(filePath, "r");
    const buf = Buffer.alloc(CHUNK_BYTES);

    const sendChunk = () => {
      const n = fs.readSync(fd, buf, 0, CHUNK_BYTES, null);
      if (n > 0) {
        // Must copy buffer because ws.send is async and we reuse buf immediately.
        ws.send(Buffer.from(buf.subarray(0, n)));
        setImmediate(sendChunk);
      } else {
        fs.closeSync(fd);
        ws.send(JSON.stringify({ type: "stop" }));
        console.log("Finished sending audio.");
      }
    };

    sendChunk();
  } catch (err) {
    console.error(`Error reading file: ${err.message}`);
    ws.close();
  }
}

const { endpoint, filePath } = parseArgs(process.argv.slice(2));
if (!endpoint || !filePath) {
  printUsageAndExit();
}

const stats = fs.statSync(filePath);
const duration = stats.size / PCM_BYTES_PER_SECOND;
const startTime = Date.now();
const ws = new WebSocket(endpoint);

console.log(`Connecting to ${endpoint}...`);
console.log(`Audio Duration: ${duration.toFixed(2)}s`);

ws.on("open", () => {
  console.log("Connected.");
  ws.send(
    JSON.stringify({
      type: "start",
      format: "pcm_s16le",
      sample_rate_hz: 16000,
      channels: 1,
    }),
  );

  streamAudio(ws, filePath);
});

ws.on("message", (data) => {
  try {
    const timestamp = new Date().toISOString().split("T")[1].slice(0, -1);
    const evt = JSON.parse(data.toString());
    if (evt.type === "partial") {
      process.stdout.write(`\r[${timestamp}] [Partial] ${evt.text}`);
    } else if (evt.type === "final") {
      console.log(`\n[${timestamp}] [Final] ${evt.text}`);
    } else {
      console.log(`\n[${timestamp}] [${evt.type}] ${JSON.stringify(evt)}`);
    }
  } catch {
    console.log("\n[Non-JSON]", data.toString());
  }
});

ws.on("close", () => {
  const processTime = (Date.now() - startTime) / 1000.0;
  const rtf = duration > 0 ? processTime / duration : 0.0;
  console.log(`\nProcessing Time: ${processTime.toFixed(2)}s`);
  console.log(`Real-Time Factor (RTF): ${rtf.toFixed(4)}`);
  console.log("\nDisconnected.");
});

ws.on("error", (err) => {
  console.error(`WebSocket error: ${err.message}`);
});
