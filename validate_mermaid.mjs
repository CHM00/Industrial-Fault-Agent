import { JSDOM } from "jsdom";

// Mermaid's parser sanitizes labels while parsing. Provide the same DOM
// primitives it expects in a browser before importing Mermaid itself.
const dom = new JSDOM("<!doctype html><html><body></body></html>");
globalThis.window = dom.window;
globalThis.document = dom.window.document;

const { default: mermaid } = await import("mermaid");
mermaid.initialize({ startOnLoad: false, securityLevel: "strict" });

// Long-lived JSON-lines protocol. Importing Mermaid/JSDOM is the expensive
// part, so the Python service keeps this process warm between validations.
process.stdin.setEncoding("utf8");
let buffer = "";
for await (const chunk of process.stdin) {
  buffer += chunk;
  let newline;
  while ((newline = buffer.indexOf("\n")) >= 0) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (!line) continue;
    let request;
    try {
      request = JSON.parse(line);
      const parsed = await mermaid.parse(String(request.source || ""));
      process.stdout.write(JSON.stringify({
        id: request.id,
        valid: true,
        diagramType: parsed?.diagramType || "flowchart",
      }) + "\n");
    } catch (error) {
      process.stdout.write(JSON.stringify({
        id: request?.id,
        valid: false,
        error: error?.message || String(error),
      }) + "\n");
    }
  }
}
