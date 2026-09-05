import React, { useCallback, useRef, useState } from "react";

const LANGS = ["auto", "en", "hi", "gu", "bn", "ur", "mr"];

// deterministic colour per token piece (tokenizer-style)
function hue(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
  return h;
}
function Tokens({ tokens }) {
  if (!tokens || !tokens.length) return null;
  return (
    <span>
      {tokens.map((t, i) => (
        <span
          key={i}
          style={{
            background: `hsl(${hue(t)} 70% 88%)`,
            color: "#111",
            borderRadius: 4,
            padding: "1px 3px",
            marginRight: 2,
            whiteSpace: "pre",
          }}
        >
          {t}
        </span>
      ))}
    </span>
  );
}

export default function App() {
  const [ip, setIp] = useState("");             // e.g. 203.0.113.5:8000
  const [lang, setLang] = useState("auto");
  const [connected, setConnected] = useState(false);
  const [listening, setListening] = useState(false);
  const [partial, setPartial] = useState({ tokens: [], text: "", language: "" });
  const [finals, setFinals] = useState([]);
  const [lat, setLat] = useState({ server: 0, e2e: 0, audio: 0 });
  const [status, setStatus] = useState("idle");

  const wsRef = useRef(null);
  const ctxRef = useRef(null);
  const nodeRef = useRef(null);
  const streamRef = useRef(null);
  const lastSendRef = useRef(0);

  const wsUrl = () => {
    let s = ip.trim().replace(/^wss?:\/\//, "").replace(/\/ws$/, "");
    return `ws://${s}/ws`;
  };

  const connect = useCallback(() => {
    if (!ip.trim()) return setStatus("enter server ip:port first");
    const ws = new WebSocket(wsUrl());
    wsRef.current = ws;
    setStatus("connecting…");
    ws.onopen = () => {
      setConnected(true);
      setStatus("connected");
      ws.send(JSON.stringify({ sample_rate: ctxRef.current?.sampleRate || 16000, lang }));
    };
    ws.onclose = () => { setConnected(false); setStatus("disconnected"); stopMic(); };
    ws.onerror = () => setStatus("ws error — check ip/port & that server is on http (ws)");
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      const e2e = lastSendRef.current ? performance.now() - lastSendRef.current : 0;
      if (m.type === "partial") {
        setPartial({ tokens: m.tokens, text: m.text, language: m.language });
        setLat({ server: m.server_ms, e2e: Math.round(e2e), audio: m.audio_ms });
      } else if (m.type === "pre_hit_llm") {
        setStatus(`⚡ prefetch-LLM [${m.language}]`);
      } else if (m.type === "end_of_speech") {
        setFinals((f) => [{ text: m.text, tokens: m.tokens, language: m.language }, ...f].slice(0, 50));
        setPartial({ tokens: [], text: "", language: "" });
        setLat({ server: m.server_ms, e2e: Math.round(e2e), audio: m.audio_ms });
      } else if (m.type === "ready") {
        setStatus(`ready · ${m.sample_rate}Hz · ${m.lang}`);
      } else if (m.type === "error") {
        setStatus("server: " + m.text);
      }
    };
  }, [ip, lang]);

  const disconnect = () => { stopMic(); wsRef.current?.close(); };

  const startMic = useCallback(async () => {
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    ctxRef.current = ctx;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    streamRef.current = stream;
    const src = ctx.createMediaStreamSource(stream);
    const node = ctx.createScriptProcessor(4096, 1, 1);   // ~256ms chunks @16kHz
    nodeRef.current = node;
    node.onaudioprocess = (ev) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== 1) return;
      const f32 = new Float32Array(ev.inputBuffer.getChannelData(0)); // copy
      lastSendRef.current = performance.now();
      ws.send(f32.buffer);
    };
    const mute = ctx.createGain();
    mute.gain.value = 0;                                   // don't echo yourself
    src.connect(node); node.connect(mute); mute.connect(ctx.destination);
    // tell server the real sample rate (browsers may ignore 16000)
    wsRef.current?.send(JSON.stringify({ sample_rate: ctx.sampleRate, lang }));
    setListening(true);
    setStatus(`listening · ${ctx.sampleRate}Hz`);
  }, [lang]);

  const stopMic = () => {
    nodeRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    ctxRef.current?.close().catch(() => {});
    nodeRef.current = null; ctxRef.current = null; streamRef.current = null;
    setListening(false);
  };

  const box = { border: "1px solid #ddd", borderRadius: 8, padding: 14, margin: "10px 0" };
  return (
    <div style={{ fontFamily: "system-ui, sans-serif", maxWidth: 820, margin: "24px auto", padding: 12 }}>
      <h2>kupe-spark-asr-270m · live streaming ASR</h2>

      <div style={box}>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <input
            style={{ flex: 1, minWidth: 220, padding: 8, fontSize: 14 }}
            placeholder="server ip:port  (e.g. 203.0.113.5:8000)"
            value={ip}
            onChange={(e) => setIp(e.target.value)}
            disabled={connected}
          />
          <select value={lang} onChange={(e) => setLang(e.target.value)}>
            {LANGS.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
          {!connected
            ? <button onClick={connect}>Connect</button>
            : <button onClick={disconnect}>Disconnect</button>}
          {connected && (listening
            ? <button onClick={stopMic}>■ Stop mic</button>
            : <button onClick={startMic}>● Start mic</button>)}
        </div>
        <div style={{ marginTop: 8, fontSize: 13, color: "#555" }}>status: {status}</div>
      </div>

      <div style={box}>
        <div style={{ display: "flex", gap: 18, fontSize: 13, color: "#333" }}>
          <span>server compute: <b>{lat.server} ms</b></span>
          <span>end-to-end: <b>{lat.e2e} ms</b></span>
          <span>chunk audio: <b>{lat.audio} ms</b></span>
        </div>
      </div>

      <div style={box}>
        <div style={{ fontSize: 12, color: "#888", marginBottom: 6 }}>
          live (partial) {partial.language && `· ${partial.language}`}
        </div>
        <div style={{ minHeight: 40, fontSize: 20, lineHeight: 1.9 }}>
          <Tokens tokens={partial.tokens} />
        </div>
      </div>

      <div style={box}>
        <div style={{ fontSize: 12, color: "#888", marginBottom: 6 }}>finalized turns</div>
        {finals.map((f, i) => (
          <div key={i} style={{ padding: "6px 0", borderBottom: "1px solid #f0f0f0" }}>
            <span style={{ fontSize: 11, color: "#999", marginRight: 6 }}>[{f.language}]</span>
            <Tokens tokens={f.tokens} />
          </div>
        ))}
      </div>
    </div>
  );
}
