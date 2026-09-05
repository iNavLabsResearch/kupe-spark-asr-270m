import React, { useCallback, useEffect, useRef, useState } from "react";

const ASR_HOST = "spark-asr.kupe.in";
const LANGS = ["auto", "en", "hi", "gu", "bn", "ur", "mr"];

function defaultServer() {
  return import.meta.env.VITE_ASR_SERVER || ASR_HOST;
}

/** Domain (nginx TLS) → wss. Raw ip:8000 on a local HTTP page → ws. */
function toWsUrl(input, pageHttps = typeof window !== "undefined" && window.location.protocol === "https:") {
  const raw = (input || "").trim();
  if (!raw) return "";
  if (/^wss?:\/\//i.test(raw)) {
    const u = raw.replace(/\/+$/, "");
    return /\/ws$/i.test(u) ? u : `${u}/ws`;
  }
  let host = raw.replace(/^https?:\/\//i, "").replace(/\/ws$/i, "").replace(/\/+$/, "");
  const ipPort = /^\d+\.\d+\.\d+\.\d+(:\d+)?$/.test(host);
  if (pageHttps && /:8000$/.test(host)) host = host.replace(/:8000$/, "");
  const secure = pageHttps || !ipPort;
  return `${secure ? "wss" : "ws"}://${host}/ws`;
}

function hue(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
  return h;
}

function Tokens({ tokens, text }) {
  if (tokens?.length) {
    return (
      <span>
        {tokens.map((t, i) => (
          <span
            key={i}
            className="token"
            style={{
              background: `hsl(${hue(t)} 40% 18%)`,
              color: `hsl(${hue(t)} 72% 82%)`,
            }}
          >
            {t}
          </span>
        ))}
      </span>
    );
  }
  return <span>{text || ""}</span>;
}

function fmtMs(n) {
  if (n == null || Number.isNaN(n)) return "—";
  return `${Math.round(n)} ms`;
}

export default function App() {
  const [ip, setIp] = useState(defaultServer);
  const [lang, setLang] = useState("auto");
  const [connected, setConnected] = useState(false);
  const [listening, setListening] = useState(false);
  const [partial, setPartial] = useState({ tokens: [], text: "", language: "" });
  const [chunks, setChunks] = useState([]);
  const [lat, setLat] = useState({ server: 0, e2e: 0, audio: 0 });
  const [status, setStatus] = useState("idle");

  const wsRef = useRef(null);
  const ctxRef = useRef(null);
  const nodeRef = useRef(null);
  const streamRef = useRef(null);
  const lastSendRef = useRef(0);
  const leftRef = useRef(null);
  const rightRef = useRef(null);
  const idRef = useRef(0);

  const wsUrl = () => toWsUrl(ip);

  const pushChunk = (m, e2e, live) => {
    const row = {
      id: ++idRef.current,
      text: m.text || "",
      tokens: m.tokens || [],
      language: m.language || "",
      server: m.server_ms,
      e2e: Math.round(e2e),
      audio: m.audio_ms,
      live,
    };
    setChunks((prev) => {
      if (prev.length && prev[prev.length - 1].live) {
        return [...prev.slice(0, -1), { ...row, id: prev[prev.length - 1].id }];
      }
      return [...prev, row];
    });
  };

  const connect = useCallback(() => {
    if (!ip.trim()) return setStatus("enter server ip:port first");
    const url = wsUrl();
    let ws;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      setStatus(err?.message || "websocket blocked — HTTPS pages need wss://");
      return;
    }
    wsRef.current = ws;
    setStatus(`connecting ${url}`);
    ws.onopen = () => {
      setConnected(true);
      setStatus("connected");
      ws.send(JSON.stringify({ sample_rate: ctxRef.current?.sampleRate || 16000, lang }));
    };
    ws.onclose = () => {
      setConnected(false);
      setStatus("disconnected");
      stopMic();
    };
    ws.onerror = () =>
      setStatus("ws error — HTTPS needs wss:// (Caddy on :443). Local HTTP can still use host:8000");
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      const e2e = lastSendRef.current ? performance.now() - lastSendRef.current : 0;
      if (m.type === "partial") {
        setPartial({ tokens: m.tokens, text: m.text, language: m.language });
        setLat({ server: m.server_ms, e2e: Math.round(e2e), audio: m.audio_ms });
        pushChunk(m, e2e, true);
      } else if (m.type === "pre_hit_llm") {
        setStatus(`prefetch-LLM [${m.language}]`);
      } else if (m.type === "end_of_speech") {
        setPartial({ tokens: [], text: "", language: "" });
        setLat({ server: m.server_ms, e2e: Math.round(e2e), audio: m.audio_ms });
        pushChunk(m, e2e, false);
      } else if (m.type === "ready") {
        setStatus(`ready · ${m.sample_rate}Hz · ${m.lang}`);
      } else if (m.type === "error") {
        setStatus("server: " + m.text);
      }
    };
  }, [ip, lang]);

  const disconnect = () => {
    stopMic();
    wsRef.current?.close();
  };

  const startMic = useCallback(async () => {
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    ctxRef.current = ctx;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    streamRef.current = stream;
    const src = ctx.createMediaStreamSource(stream);
    const node = ctx.createScriptProcessor(4096, 1, 1);
    nodeRef.current = node;
    node.onaudioprocess = (ev) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== 1) return;
      const f32 = new Float32Array(ev.inputBuffer.getChannelData(0));
      lastSendRef.current = performance.now();
      ws.send(f32.buffer);
    };
    const mute = ctx.createGain();
    mute.gain.value = 0;
    src.connect(node);
    node.connect(mute);
    mute.connect(ctx.destination);
    wsRef.current?.send(JSON.stringify({ sample_rate: ctx.sampleRate, lang }));
    setListening(true);
    setStatus(`listening · ${ctx.sampleRate}Hz`);
  }, [lang]);

  const stopMic = () => {
    nodeRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    ctxRef.current?.close().catch(() => {});
    nodeRef.current = null;
    ctxRef.current = null;
    streamRef.current = null;
    setListening(false);
  };

  const clearChat = () => {
    setChunks([]);
    setPartial({ tokens: [], text: "", language: "" });
    setLat({ server: 0, e2e: 0, audio: 0 });
    idRef.current = 0;
  };

  const finals = chunks.filter((c) => !c.live);
  const fullText = [finals.map((c) => c.text).filter(Boolean).join(" "), partial.text]
    .filter(Boolean)
    .join(" ")
    .trim();

  useEffect(() => {
    if (leftRef.current) leftRef.current.scrollTop = leftRef.current.scrollHeight;
    if (rightRef.current) rightRef.current.scrollTop = rightRef.current.scrollHeight;
  }, [chunks, partial.text]);

  const dotClass = connected ? "on" : status.includes("connecting") ? "wait" : "off";

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="/brand/kupe-mark.svg" alt="Kupe" />
          <div className="brand-copy">
            <span className="wordmark">kupe</span>
            <span className="product">Spark ASR 270M · live</span>
          </div>
        </div>

        <div className="controls">
          <span className={`dot ${dotClass}`} title={status} />
          <input
            className="field server"
            placeholder="spark-asr.kupe.in"
            value={ip}
            onChange={(e) => setIp(e.target.value)}
            disabled={connected}
          />
          <select className="field lang" value={lang} onChange={(e) => setLang(e.target.value)}>
            {LANGS.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
          {!connected ? (
            <button className="btn" onClick={connect}>
              Connect
            </button>
          ) : (
            <button className="btn btn-danger" onClick={disconnect}>
              Disconnect
            </button>
          )}
          {connected &&
            (listening ? (
              <button className="btn btn-stop" onClick={stopMic}>
                Stop mic
              </button>
            ) : (
              <button className="btn btn-mic" onClick={startMic}>
                Start mic
              </button>
            ))}
          <button className="btn btn-ghost" onClick={clearChat} disabled={!chunks.length && !partial.text}>
            Clear
          </button>
        </div>
      </header>

      <div className="stats">
        <span>
          server <b>{fmtMs(lat.server)}</b>
        </span>
        <span>
          end-to-end <b>{fmtMs(lat.e2e)}</b>
        </span>
        <span>
          chunk audio <b>{fmtMs(lat.audio)}</b>
        </span>
        <span className="status">{status}</span>
      </div>

      <div className="workspace">
        <section className="pane">
          <div className="pane-head">
            <h2>Chunks</h2>
            <span className="pane-count">{chunks.length}</span>
          </div>
          <div className="scroll" ref={leftRef}>
            {!chunks.length && <div className="empty">Speak after connecting — each decode lands here with latency.</div>}
            {chunks.map((c, i) => (
              <div key={c.id} className={`chunk${c.live ? " live" : ""}`}>
                <span className="chunk-n">{i + 1}</span>
                <div className="chunk-body">
                  <div className="chunk-meta">
                    {c.language && <span className="lang-tag">{c.language}</span>}
                    {c.live && <span className="live-tag">live</span>}
                  </div>
                  <Tokens tokens={c.tokens} text={c.text} />
                </div>
                <div className="lat">
                  <span className="lat-ms">{fmtMs(c.server)}</span>
                  <span className="lat-sub">e2e {fmtMs(c.e2e)}</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="pane">
          <div className="pane-head">
            <h2>Full transcript</h2>
            <button className="btn btn-ghost" onClick={clearChat} disabled={!chunks.length && !partial.text}>
              Clear chat
            </button>
          </div>
          <div className="scroll" ref={rightRef}>
            {!fullText ? (
              <div className="empty">Accumulated transcript will appear here.</div>
            ) : (
              <div className="full">
                {finals.map((c) => c.text).filter(Boolean).join(" ")}
                {partial.text ? (
                  <>
                    {finals.some((c) => c.text) ? " " : ""}
                    <span className="live-text">{partial.text}</span>
                  </>
                ) : null}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
