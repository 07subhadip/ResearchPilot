"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import {
    BookOpen,
    Clock,
    Zap,
    AlertCircle,
    CheckCircle,
    ExternalLink,
    Sparkles,
    Brain,
    ArrowRight,
    Layers,
    Fingerprint,
    Send,
    Info,
    X,
    Server,
    Activity,
    Rocket,
} from "lucide-react";

// ── Types ─────────────────────────────────────────────────
interface Citation {
    paper_id: string;
    title: string;
    authors: string[];
    published_date: string;
    arxiv_url: string;
}

interface QueryResult {
    answer: string;
    citations: Citation[];
    query: string;
    chunks_used: number;
    retrieval_time_ms: number;
    generation_time_ms: number;
    total_time_ms: number;
    has_context: boolean;
}

// ── Config ────────────────────────────────────────────────
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const EXAMPLE_QUERIES = [
    "How does LoRA reduce trainable parameters?",
    "What are challenges in multi-agent RL?",
    "Explain diffusion models for images",
];

const CATEGORY_OPTIONS = [
    { value: "All", label: "All Topics" },
    { value: "cs.LG", label: "cs.LG", indexed: true },
    { value: "cs.AI", label: "cs.AI", indexed: true },
    { value: "stat.ML", label: "stat.ML", indexed: true },
    { value: "cs.CV", label: "cs.CV", indexed: false, disabled: true },
    { value: "cs.CL", label: "cs.CL", indexed: false, disabled: true },
    { value: "cs.RO", label: "cs.RO", indexed: false, disabled: true },
];

// ── Custom Dropdown Component ─────────────────────────────
function CustomSelect({
    options,
    value,
    onChange,
    width = '140px',
}: {
    options: { value: string | number; label: string; disabled?: boolean; indexed?: boolean }[];
    value: string | number;
    onChange: (val: string | number) => void;
    width?: string;
}) {
    const [isOpen, setIsOpen] = useState(false);
    const [placement, setPlacement] = useState<"top" | "bottom">("bottom");
    const ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (ref.current && !ref.current.contains(e.target as Node)) setIsOpen(false);
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const toggleOpen = () => {
        if (!isOpen && ref.current) {
            const rect = ref.current.getBoundingClientRect();
            // Need ~240px for full menu, pop up if space is tight below
            if (window.innerHeight - rect.bottom < 240) {
                setPlacement('top');
            } else {
                setPlacement('bottom');
            }
        }
        setIsOpen(!isOpen);
    };

    const activeLabel = options.find((o) => o.value === value)?.label || value;

    return (
        <div ref={ref} style={{ position: 'relative', width }}>
            <button
                onClick={toggleOpen}
                className="cyber-select"
                style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{activeLabel}</span>
                <span style={{ fontSize: '0.7em', opacity: 0.5, marginLeft: '8px' }}>{isOpen ? '▲' : '▼'}</span>
            </button>
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: placement === 'top' ? 10 : -10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: placement === 'top' ? 5 : -5, scale: 0.95 }}
                        transition={{ duration: 0.15, ease: 'easeOut' }}
                        className="custom-dropdown-menu"
                        style={{
                            top: placement === 'bottom' ? 'calc(100% + 8px)' : 'auto',
                            bottom: placement === 'top' ? 'calc(100% + 8px)' : 'auto'
                        }}
                    >
                        {options.map((opt) => (
                            <button
                                key={opt.value}
                                onClick={() => {
                                    if (opt.disabled) return;
                                    onChange(opt.value);
                                    setIsOpen(false);
                                }}
                                disabled={opt.disabled}
                                className={`custom-dropdown-item ${value === opt.value ? 'active' : ''}`}
                                style={opt.disabled ? { opacity: 0.4, cursor: 'not-allowed' } : {}}
                            >
                                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", width: "100%" }}>
                                    <span>{opt.label}</span>
                                    {opt.indexed !== undefined && (
                                        <div style={{
                                            fontSize: "0.65em",
                                            fontWeight: 600,
                                            padding: "2px 6px",
                                            borderRadius: "12px",
                                            backgroundColor: opt.indexed ? "rgba(16, 185, 129, 0.15)" : "rgba(156, 163, 175, 0.1)",
                                            color: opt.indexed ? "var(--success)" : "var(--text-muted)",
                                            marginLeft: "8px"
                                        }}>
                                            {opt.indexed ? "INDEXED" : "UNAVAILABLE"}
                                        </div>
                                    )}
                                </div>
                            </button>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

function renderLaTeX(text: string) {
    const blockSplit = text.split(/(\$\$[\s\S]+?\$\$)/g);
    return blockSplit.map((blockPart, i) => {
        if (blockPart.startsWith("$$") && blockPart.endsWith("$$")) {
            const math = blockPart.slice(2, -2);
            return <BlockMath key={i} math={math} />;
        }
        const inlineSplit = blockPart.split(/(\$[\s\S]+?\$)/g);
        return (
            <span key={i}>
                {inlineSplit.map((inlinePart, j) => {
                    if (inlinePart.startsWith("$") && inlinePart.endsWith("$")) {
                        const math = inlinePart.slice(1, -1);
                        return <InlineMath key={j} math={math} />;
                    }
                    return (
                        <span key={j} style={{ whiteSpace: "pre-wrap" }}>
                            {inlinePart}
                        </span>
                    );
                })}
            </span>
        );
    });
}

// ── Main Page ─────────────────────────────────────────────
export default function Home() {
    const [question, setQuestion] = useState("");
    const [result, setResult] = useState<QueryResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    // Settings
    const [topK, setTopK] = useState(5);
    const [category, setCategory] = useState("All");
    const [yearFilter, setYearFilter] = useState(false);
    const [yearFrom, setYearFrom] = useState(2024);
    const [showSettings, setShowSettings] = useState(false);

    // System state
    const [apiStatus, setApiStatus] = useState<
        "unknown" | "online" | "offline"
    >("unknown");
    const [showInfo, setShowInfo] = useState(false);
    const [showStatusDetails, setShowStatusDetails] = useState(false);

    const textareaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
        }
    }, [question]);

    useEffect(() => {
        checkHealth();
    }, []);

    const checkHealth = async () => {
        try {
            const r = await fetch(`${API_URL}/health`, {
                signal: AbortSignal.timeout(5000),
            });
            setApiStatus(r.ok ? "online" : "offline");
        } catch {
            setApiStatus("offline");
        }
    };

    const handleSearch = async () => {
        if (!question.trim()) return;
        setLoading(true);
        setError("");
        setResult(null);

        try {
            const payload: Record<string, unknown> = {
                question: question.trim(),
                top_k: topK,
            };
            if (category !== "All") payload.filter_category = category;
            if (yearFilter) payload.filter_year_gte = yearFrom;

            const response = await fetch(`${API_URL}/query`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!response.ok)
                throw new Error(`API returned ${response.status}`);
            const data: QueryResult = await response.json();
            setResult(data);
        } catch (err) {
            setError(
                err instanceof Error
                    ? err.message
                    : "Failed to connect to API.",
            );
        } finally {
            setLoading(false);
        }
    };

    const resetApp = () => {
        setQuestion("");
        setResult(null);
        setError("");
        setLoading(false);
    };

    const hasSearched = result || loading || error;

    return (
        <>
            <div className="luminous-grid" />
            <div className="orb orb-cyan" />
            <div className="orb orb-purple" />

            <AnimatePresence>
                {showInfo && (
                    <div className="info-modal-backdrop" onClick={() => setShowInfo(false)}>
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 20 }}
                            onClick={(e) => e.stopPropagation()}
                            className="cyber-panel info-modal"
                        >
                            <button className="modal-close" onClick={() => setShowInfo(false)}>
                                <X size={18} />
                            </button>
                            <h2>ResearchPilot Console</h2>
                            <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "16px" }}>
                                <div style={{ background: "rgba(138, 43, 226, 0.15)", border: "1px solid rgba(138, 43, 226, 0.4)", padding: "6px 14px", borderRadius: "99px", fontSize: "0.75rem", color: "var(--accent-2)", fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase" }}>
                                    Lead Architect
                                </div>
                                <span style={{ fontFamily: "'Dancing Script', cursive", fontSize: "1.8rem", fontWeight: 700, background: "linear-gradient(135deg, #fff 20%, var(--accent-2) 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", letterSpacing: "0.05em", transform: "translateY(-2px)" }}>Subhadip Hensh</span>
                            </div>
                            
                            <hr />
                            
                            <h3><Server size={18} /> System Overview</h3>
                            <p style={{ fontSize: "1rem", lineHeight: 1.7, marginBottom: "16px" }}>ResearchPilot is a high-performance RAG (Retrieval-Augmented Generation) engine tailored for Machine Learning literature. It features hybrid sparse-dense searching, advanced cross-encoder reranking, and GPU-driven vector indexing via Qdrant.</p>
                            
                            <h3><Activity size={18} /> Current Operational Capacity</h3>
                            <ul>
                                <li><strong>Current Index</strong> Synthesizing 51,019 dense embeddings isolated from ~700 major AI & ML papers.</li>
                                <li><strong>Data Categories</strong> Fully indexed on core Machine Learning (cs.LG) and AI (cs.AI).</li>
                            </ul>

                            <h3><Layers size={18} /> Core Technology Stack</h3>
                            <ul>
                                <li><strong>Frontend Application</strong> Next.js 16 (App Router), React, Framer Motion, Vanilla CSS (Glassmorphism).</li>
                                <li><strong>Backend Environment</strong> Python, FastAPI, Uvicorn, Pydantic.</li>
                                <li><strong>Vector Database Engine</strong> Qdrant (GPU Accelerated Dense Vectors).</li>
                                <li><strong>RAG Processing Pipeline</strong> SentenceTransformers (BGE-base), BM25 Sparse Search, Cross-Encoder Reranking, Groq LLM (LLaMA 3.3).</li>
                                <li><strong>Mathematics Engine</strong> KaTeX & React-KaTeX for fully dynamic native LaTeX equations.</li>
                            </ul>

                            <h3><Rocket size={18} /> Phase 2: In-Progress Architecture</h3>
                            <ul>
                                <li><strong>Massive Data Expansion</strong> Scaling dataset soon to 10,000+ — 20,000+ ML papers spanning NLP, Computer Vision, and Robotics.</li>
                                <li><strong>Distributed Hardware Execution</strong> Scaling ingestion logic to cloud-based GPU clusters for extreme speed.</li>
                                <li><strong>Multi-modal Analysis</strong> Soon integrating visual graph and chart processing abilities into the synthesis engine.</li>
                            </ul>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>

            {/* ── Top Nav ── */}
            <header className="top-nav">
                    <div className="brand" onClick={resetApp} style={{ cursor: "pointer" }}>
                        <div className="brand-icon">
                            <Brain size={22} />
                        </div>
                        <span>ResearchPilot</span>
                    </div>
                    <div className="nav-right">
                        <button onClick={() => setShowInfo(true)} className="nav-icon-btn" aria-label="Project Info">
                            <Info size={16} />
                        </button>
                        <div style={{ position: "relative" }}>
                            <button 
                                onClick={async () => {
                                    await checkHealth();
                                    setShowStatusDetails(!showStatusDetails);
                                }} 
                                className="nav-status"
                            >
                                <div
                                    className={`status-dot ${apiStatus === "online" ? "status-online" : apiStatus === "offline" ? "status-offline" : ""}`}
                                />
                                {apiStatus === "online"
                                    ? "Systems Nominal"
                                    : apiStatus === "offline"
                                      ? "Offline"
                                      : "Checking..."}
                            </button>

                            <AnimatePresence>
                                {showStatusDetails && (
                                    <motion.div
                                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                                        animate={{ opacity: 1, y: 0, scale: 1 }}
                                        exit={{ opacity: 0, y: 10, scale: 0.95 }}
                                        className="cyber-panel custom-dropdown-menu"
                                        style={{
                                            position: "absolute",
                                            top: "calc(100% + 12px)",
                                            right: "-40px",
                                            left: "auto",
                                            width: "260px",
                                            padding: "16px",
                                            zIndex: 100,
                                            display: "flex",
                                            flexDirection: "column",
                                            gap: "8px",
                                            cursor: "default"
                                        }}
                                        onClick={(e) => e.stopPropagation()}
                                    >
                                        <h4 style={{ fontSize: "0.85rem", color: "#fff", marginBottom: "4px" }}>System Connection Status</h4>
                                        <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
                                            {apiStatus === "online" 
                                                ? "🟢 Backend API and Qdrant Vector Database are connected and responding correctly. The system is ready for inference."
                                                : "🔴 Backend API is unreachable. You need to run 'python run_api.py' in your backend directory to enable RAG functionality."}
                                        </p>
                                        <button 
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                checkHealth();
                                            }}
                                            style={{
                                                marginTop: "8px",
                                                fontSize: "0.75rem",
                                                color: "var(--accent)",
                                                background: "rgba(0, 240, 255, 0.1)",
                                                padding: "6px 12px",
                                                borderRadius: "8px",
                                                border: "1px solid rgba(0, 240, 255, 0.3)",
                                                textAlign: "center",
                                                display: "block",
                                                width: "100%",
                                                cursor: "pointer",
                                                transition: "0.2s"
                                            }}
                                            onMouseEnter={(e) => {
                                                e.currentTarget.style.background = "rgba(0, 240, 255, 0.2)";
                                            }}
                                            onMouseLeave={(e) => {
                                                e.currentTarget.style.background = "rgba(0, 240, 255, 0.1)";
                                            }}
                                        >
                                            Re-verify Connection
                                        </button>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                        <a 
                            href="https://github.com/07subhadip" 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            className="github-link"
                            aria-label="GitHub Profile"
                        >
                            <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
                                <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
                            </svg>
                        </a>
                    </div>
                </header>

            <main className="app-container">
                {/* ── Central Hero Block ── */}
                <motion.div
                    layout
                    className="search-wrapper"
                    style={{ marginTop: hasSearched ? "130px" : "15vh" }}
                    transition={{ type: "spring", bounce: 0.2, duration: 0.8 }}
                >
                    <AnimatePresence>
                        {!hasSearched && (
                            <motion.div
                                layout
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                transition={{ duration: 0.5 }}
                                style={{ width: "100%" }}
                            >
                                <h1 className="hero-title">
                                    Decipher the latest
                                    <br />
                                    <span className="text-gradient-2">
                                        ML Research
                                    </span>
                                </h1>
                                <p className="hero-subtitle">
                                    Neural hybrid search across ArXiv. <br />
                                    Cross-encoder reranked. LLM synthesized.
                                </p>
                                <div style={{ height: "64px" }} />
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* ── ChatGPT Style Search Box ── */}
                    <motion.div
                        layout
                        style={{
                            width: "100%",
                            margin: "0 auto",
                            zIndex: 20,
                        }}
                    >
                        <div className="chat-input-wrapper">
                            <textarea
                                ref={textareaRef}
                                value={question}
                                onChange={(e) => setQuestion(e.target.value)}
                                placeholder="Message ResearchPilot..."
                                rows={1}
                                className="chat-input"
                                style={{
                                    minHeight: hasSearched ? "20px" : "28px",
                                    maxHeight: "120px",
                                    overflowY: "auto",
                                }}
                                onKeyDown={(e) => {
                                    if (e.key === "Enter" && !e.shiftKey) {
                                        e.preventDefault();
                                        handleSearch();
                                    }
                                }}
                            />
                            <button
                                onClick={handleSearch}
                                disabled={loading || !question.trim()}
                                className="send-btn"
                            >
                                {loading ? (
                                    <div className="spinner-micro" />
                                ) : (
                                    <Send
                                        size={18}
                                        strokeWidth={2.5}
                                        style={{ marginLeft: "-2px" }}
                                    />
                                )}
                            </button>
                        </div>

                        <div className="search-controls">
                            <button
                                onClick={() => setShowSettings(!showSettings)}
                                className="controls-group"
                                style={{
                                    color: "var(--text-muted)",
                                    fontSize: "0.8rem",
                                    fontWeight: 600,
                                    padding: "4px",
                                    cursor: "pointer",
                                }}
                            >
                                <Layers size={14} />
                                CONFIGURE {showSettings ? "▲" : "▼"}
                            </button>

                            <div className="controls-group">
                                <AnimatePresence>
                                    {showSettings && (
                                        <motion.div
                                            initial={{ opacity: 0, height: 0 }}
                                            animate={{
                                                opacity: 1,
                                                height: "auto",
                                            }}
                                            exit={{ opacity: 0, height: 0 }}
                                            style={{
                                                overflow: "visible",
                                                display: "flex",
                                                gap: "12px",
                                                flexWrap: "wrap",
                                                alignItems: "center",
                                            }}
                                        >
                                            <CustomSelect
                                                value={topK}
                                                onChange={(val) => setTopK(Number(val))}
                                                options={[
                                                    { value: 3, label: 'Top 3 Results' },
                                                    { value: 5, label: 'Top 5 Results' },
                                                    { value: 10, label: 'Top 10 Results' },
                                                ]}
                                            />
                                            <CustomSelect
                                                value={category}
                                                onChange={(val) => setCategory(String(val))}
                                                options={CATEGORY_OPTIONS}
                                                width="160px"
                                            />
                                            <button
                                                onClick={() =>
                                                    setYearFilter(!yearFilter)
                                                }
                                                className={`cyber-btn-outline ${yearFilter ? "active" : ""}`}
                                            >
                                                YEAR FILTER{" "}
                                                {yearFilter ? "ON" : "OFF"}
                                            </button>
                                            {yearFilter && (
                                                <div className="year-stepper">
                                                    <button onClick={() => setYearFrom(y => Math.max(2000, y - 1))} className="stepper-btn">-</button>
                                                    <input
                                                        type="number"
                                                        value={yearFrom}
                                                        readOnly
                                                        className="stepper-input"
                                                    />
                                                    <button onClick={() => setYearFrom(y => Math.min(2026, y + 1))} className="stepper-btn">+</button>
                                                </div>
                                            )}
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        </div>
                    </motion.div>

                    {/* ── Example Queries ── */}
                    <AnimatePresence>
                        {!hasSearched && (
                            <motion.div
                                layout
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0, filter: "blur(5px)" }}
                                className="example-chips"
                            >
                                {EXAMPLE_QUERIES.map((q, i) => (
                                    <motion.button
                                        key={q}
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: 0.1 * i + 0.3 }}
                                        onClick={() => {
                                            setQuestion(q);
                                            setTimeout(
                                                () => handleSearch(),
                                                50,
                                            );
                                        }}
                                        className="chip"
                                    >
                                        {q}
                                    </motion.button>
                                ))}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </motion.div>

                {/* ── Error State ── */}
                <AnimatePresence>
                    {error && (
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="cyber-panel"
                            style={{
                                width: "100%",
                                padding: "24px",
                                borderColor: "var(--danger)",
                                marginTop: "24px",
                            }}
                        >
                            <div
                                style={{
                                    display: "flex",
                                    gap: "16px",
                                    alignItems: "center",
                                }}
                            >
                                <AlertCircle size={32} color="var(--danger)" />
                                <div>
                                    <h3
                                        style={{
                                            color: "var(--danger)",
                                            fontSize: "1.2rem",
                                            marginBottom: "4px",
                                        }}
                                    >
                                        Critical Exception
                                    </h3>
                                    <p style={{ color: "var(--text-muted)" }}>
                                        {error}
                                    </p>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* ── Loading View ── */}
                <AnimatePresence>
                    {loading && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="loader-view"
                            style={{ width: "100%" }}
                        >
                            <div className="ring-spinner">
                                <div className="ring ring-1" />
                                <div className="ring ring-2" />
                                <Brain
                                    size={22}
                                    style={{
                                        position: "absolute",
                                        top: "24px",
                                        left: "24px",
                                        color: "var(--text-main)",
                                    }}
                                />
                            </div>
                            <div>
                                <h2
                                    style={{
                                        fontSize: "1.4rem",
                                        fontWeight: 600,
                                        color: "#fff",
                                        marginBottom: "8px",
                                    }}
                                >
                                    Synthesizing Knowledge
                                </h2>
                                <p
                                    style={{
                                        color: "var(--text-muted)",
                                        fontSize: "0.95rem",
                                    }}
                                >
                                    Running Vector Search & LLM Inference
                                </p>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* ── Results Output ── */}
                {result && !loading && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ staggerChildren: 0.1 }}
                        className="results-area"
                    >
                        <motion.div
                            className="cyber-panel answer-box"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                        >
                            <div className="answer-header">
                                <div className="answer-label">
                                    <Sparkles size={20} /> AI Synthesis
                                </div>
                                {result.has_context ? (
                                    <div className="badge-grounded">
                                        <CheckCircle size={14} /> Grounded
                                        Sources
                                    </div>
                                ) : (
                                    <div
                                        className="badge-grounded"
                                        style={{
                                            color: "var(--danger)",
                                            borderColor:
                                                "rgba(239, 68, 68, 0.3)",
                                            background:
                                                "rgba(239, 68, 68, 0.1)",
                                        }}
                                    >
                                        <AlertCircle size={14} /> Hallucination
                                        Risk
                                    </div>
                                )}
                            </div>
                            <div className="answer-text">{renderLaTeX(result.answer)}</div>
                        </motion.div>

                        <motion.div
                            className="stats-grid"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                        >
                            {[
                                {
                                    l: "Execution Time",
                                    v: `${(result.total_time_ms / 1000).toFixed(1)}s`,
                                    i: Clock,
                                    c: "var(--accent)",
                                },
                                {
                                    l: "Vector Data",
                                    v: `${(result.retrieval_time_ms / 1000).toFixed(1)}s`,
                                    i: ArrowRight,
                                    c: "#fff",
                                },
                                {
                                    l: "LLM Generation",
                                    v: `${(result.generation_time_ms / 1000).toFixed(1)}s`,
                                    i: Zap,
                                    c: "var(--accent-2)",
                                },
                                {
                                    l: "Paper Chunks",
                                    v: result.chunks_used,
                                    i: Fingerprint,
                                    c: "var(--success)",
                                },
                            ].map((s, i) => (
                                <div key={i} className="cyber-panel stat-card">
                                    <div
                                        className="stat-header"
                                        style={{ width: "100%" }}
                                    >
                                        <span className="stat-label">
                                            {s.l}
                                        </span>
                                        <s.i
                                            size={16}
                                            color={s.c}
                                            style={{ opacity: 0.8 }}
                                        />
                                    </div>
                                    <div
                                        className="stat-value"
                                        style={{
                                            width: "100%",
                                            textAlign: "left",
                                            color: s.c,
                                        }}
                                    >
                                        {s.v}
                                    </div>
                                </div>
                            ))}
                        </motion.div>

                        {result.citations.length > 0 && (
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                            >
                                <div className="section-title">
                                    <BookOpen
                                        size={18}
                                        color="var(--accent-2)"
                                    />
                                    Extracted Literature
                                </div>
                                <div className="citations-grid">
                                    {result.citations.map((cite, i) => (
                                        <a
                                            href={cite.arxiv_url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            key={i}
                                            className="cyber-panel citation-card"
                                        >
                                            <div className="citation-open">
                                                <ExternalLink size={16} />
                                            </div>
                                            <div className="citation-meta">
                                                <span className="citation-id">
                                                    {cite.paper_id}
                                                </span>
                                                <span className="citation-date">
                                                    {cite.published_date}
                                                </span>
                                            </div>
                                            <h4 className="citation-title">
                                                {cite.title}
                                            </h4>
                                            <div className="citation-authors">
                                                {cite.authors
                                                    .slice(0, 3)
                                                    .join(", ")}
                                                {cite.authors.length > 3 &&
                                                    ` +${cite.authors.length - 3} more`}
                                            </div>
                                        </a>
                                    ))}
                                </div>
                            </motion.div>
                        )}
                    </motion.div>
                )}
            </main>
        </>
    );
}
