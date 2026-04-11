'use client';

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { InlineMath, BlockMath } from 'react-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
    Brain, Search, PanelLeftClose, PanelLeft, Plus,
    Send, Settings2, Trash2, Copy, Check, Star, ThumbsUp, ThumbsDown
} from "lucide-react";

// Config
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// --- Types ---
interface Citation {
    paper_id: string;
    title: string;
    authors: string[];
    published_date: string;
    arxiv_url: string;
}

interface MessageTiming {
    retrieval_time_ms?: number;
    generation_time_ms?: number;
    total_time_ms?: number;
    chunks_used?: number;
}

interface ChatMessage {
    id: string;
    role: "user" | "assistant";
    content: string;
    citations?: Citation[];
    timing?: MessageTiming;
    model_used?: string;
    timestamp: number;
}

interface ChatSession {
    id: string;
    title: string;
    messages: ChatMessage[];
    timestamp: number;
}

// --- Message Renderer Component ---
const CodeBlock = ({ language, code }: { language: string, code: string }) => {
    const [copied, setCopied] = useState(false);
    const handleCopy = () => {
        navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };
    return (
        <div className="code-wrapper">
            <div className="code-header">
                <span>{language || 'text'}</span>
                <button onClick={handleCopy} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {copied ? <><Check size={14} /> Copied!</> : <><Copy size={14} /> Copy</>}
                </button>
            </div>
            <SyntaxHighlighter language={language || 'text'} style={oneDark} customStyle={{ margin: 0, borderTopLeftRadius: 0, borderTopRightRadius: 0 }}>
                {code}
            </SyntaxHighlighter>
        </div>
    );
};

const CitationBadge = ({ id }: { id: string }) => {
    return (
        <a 
            href={`https://arxiv.org/abs/${id}`} 
            target="_blank" 
            rel="noopener noreferrer" 
            className="citation-badge"
            title={`View paper ${id} on ArXiv`}
        >
            {id}
        </a>
    );
};

// Safe rendering to avoid KaTeX crashing
const SafeMathBlock = ({ math }: { math: string }) => {
    try { return <BlockMath math={math} />; } catch (e) { return <pre>{`$$${math}$$`}</pre>; }
};
const SafeMathInline = ({ math }: { math: string }) => {
    try { return <InlineMath math={math} />; } catch (e) { return <span>{`$${math}$`}</span>; }
};

const renderTextWithMathAndCitations = (text: string) => {
    // 1. Block Math
    const blockSplit = text.split(/(\$\$[\s\S]+?\$\$)/g);
    return blockSplit.map((bPart, i) => {
        if (bPart.startsWith("$$") && bPart.endsWith("$$")) {
            return <SafeMathBlock key={i} math={bPart.slice(2, -2)} />;
        }
        
        // 2. Inline Math
        const inlineSplit = bPart.split(/(\$[^\$\n]+?\$)/g);
        return inlineSplit.map((iPart, j) => {
            if (iPart.startsWith("$") && iPart.endsWith("$")) {
                return <SafeMathInline key={`${i}-${j}`} math={iPart.slice(1, -1)} />;
            }

            // 3. Citations
            const citeSplit = iPart.split(/\[(\d{4}\.\d{4,5})\]/g);
            return citeSplit.map((cPart, k) => {
                if (/^\d{4}\.\d{4,5}$/.test(cPart)) {
                    return <CitationBadge key={`${i}-${j}-${k}`} id={cPart} />;
                }
                return <span key={`${i}-${j}-${k}`} style={{ whiteSpace: "pre-wrap" }}>{cPart}</span>;
            });
        });
    });
};

const MessageRenderer = ({ content }: { content: string }) => {
    // Split by code blocks first
    const parts = content.split(/(```[\s\S]*?```)/g);
    return (
        <div style={{ lineHeight: 1.6 }}>
            {parts.map((part, i) => {
                if (part.startsWith('```') && part.endsWith('```')) {
                    const match = part.match(/```(\w+)?\n([\s\S]*?)```/);
                    if (match) {
                        return <CodeBlock key={i} language={match[1]} code={match[2]} />;
                    }
                }
                return <div key={i}>{renderTextWithMathAndCitations(part)}</div>;
            })}
        </div>
    );
};

// --- Feedback Component ---
const FeedbackRow = ({ query, time, citationsCount, model }: { query: string, time: number, citationsCount: number, model: string }) => {
    const [rating, setRating] = useState(0);
    const [hoverRating, setHoverRating] = useState(0);
    const [thumbs, setThumbs] = useState<"up"|"down"|null>(null);
    const [comment, setComment] = useState("");
    const [submitted, setSubmitted] = useState(false);

    const handleSubmit = async () => {
        try {
            await fetch(`${API_URL}/feedback`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query, rating, thumbs, comment, model_used: model, citations_count: citationsCount, total_time_ms: time })
            });
        } catch (e) {
            console.error(e);
        }
        setSubmitted(true);
    };

    if (submitted) return <div style={{ fontSize: "0.85rem", color: "var(--success)", marginTop: "12px" }}><Check size={14} style={{ display: 'inline', marginRight: '4px' }} />Thank you for your feedback!</div>;

    return (
        <div className="feedback-row">
            <div style={{ display: "flex", gap: "4px" }}>
                {[1,2,3,4,5].map(star => (
                    <Star 
                        key={star} 
                        size={18} 
                        className={`star-btn ${(hoverRating || rating) >= star ? 'active' : ''}`}
                        onMouseEnter={() => setHoverRating(star)}
                        onMouseLeave={() => setHoverRating(0)}
                        onClick={() => setRating(star)}
                        fill={(hoverRating || rating) >= star ? "#f59e0b" : "transparent"}
                    />
                ))}
            </div>
            <div style={{ display: "flex", gap: "8px", borderLeft: "1px solid rgba(255,255,255,0.1)", paddingLeft: "12px" }}>
                <ThumbsUp size={18} className={`star-btn ${thumbs === 'up' ? 'active' : ''}`} onClick={() => setThumbs('up')} />
                <ThumbsDown size={18} className={`star-btn ${thumbs === 'down' ? 'active' : ''}`} onClick={() => setThumbs('down')} />
            </div>
            {(rating > 0 || thumbs) && (
                <>
                    <input 
                        type="text" 
                        className="feedback-input" 
                        placeholder="Tell us more (optional)" 
                        value={comment} 
                        onChange={(e) => setComment(e.target.value)} 
                    />
                    <button className="feedback-submit" onClick={handleSubmit}>Submit</button>
                </>
            )}
        </div>
    );
};

export default function App() {
    const [sessions, setSessions] = useState<ChatSession[]>([]);
    const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
    const [query, setQuery] = useState("");
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [topK, setTopK] = useState(5);
    const [category, setCategory] = useState("All");

    const chatEndRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // Load from local storage
    useEffect(() => {
        const stored = localStorage.getItem("rp_history");
        if (stored) {
            setSessions(JSON.parse(stored));
        }
    }, []);

    // Save to local storage
    useEffect(() => {
        if (sessions.length > 0) {
            localStorage.setItem("rp_history", JSON.stringify(sessions));
        }
    }, [sessions]);

    // Auto resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
        }
    }, [query]);

    // Scroll to bottom
    const scrollToBottom = () => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [sessions, activeSessionId, isStreaming]);

    const activeSession = sessions.find(s => s.id === activeSessionId);
    const currentMessages = activeSession?.messages || [];

    const handleNewChat = () => {
        setActiveSessionId(null);
        setSidebarOpen(false);
    };

    const handleSend = async () => {
        if (!query.trim() || isStreaming) return;
        
        let sessionId = activeSessionId;
        if (!sessionId) {
            sessionId = Date.now().toString();
            const newSession: ChatSession = {
                id: sessionId,
                title: query.trim().substring(0, 50) + (query.length > 50 ? "..." : ""),
                messages: [],
                timestamp: Date.now()
            };
            setSessions(prev => [newSession, ...prev]);
            setActiveSessionId(sessionId);
        }

        const userMessage: ChatMessage = {
            id: Date.now().toString(),
            role: "user",
            content: query.trim(),
            timestamp: Date.now()
        };

        const aiMessageId = (Date.now() + 1).toString();
        const placeholderAiMessage: ChatMessage = {
            id: aiMessageId,
            role: "assistant",
            content: "",
            timestamp: Date.now() + 1
        };

        setSessions(prev => prev.map(s => {
            if (s.id === sessionId) {
                return { ...s, messages: [...s.messages, userMessage, placeholderAiMessage] };
            }
            return s;
        }));

        const originalQuery = query;
        setQuery("");
        setIsStreaming(true);

        try {
            const res = await fetch(`${API_URL}/query/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    question: originalQuery, 
                    top_k: topK, 
                    filter_category: category === "All" ? undefined : category,
                    filter_year_gte: yearFilter ? yearFrom : undefined
                })
            });

            if (!res.ok || !res.body) {
                const errText = await res.text();
                throw new Error(`API error: ${res.status} ${errText}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            
            let accumulatedContent = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split("\n");

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.done) {
                                // Final update with citations + timing
                                setSessions(prev => prev.map(s => {
                                    if (s.id === sessionId) {
                                        return {
                                            ...s,
                                            messages: s.messages.map(m => 
                                                m.id === aiMessageId 
                                                ? { ...m, citations: data.citations, timing: data.timing, model_used: data.model_used }
                                                : m
                                            )
                                        };
                                    }
                                    return s;
                                }));
                            } else if (data.token) {
                                accumulatedContent += data.token;
                                // Update streaming content
                                setSessions(prev => prev.map(s => {
                                    if (s.id === sessionId) {
                                        return {
                                            ...s,
                                            messages: s.messages.map(m => 
                                                m.id === aiMessageId 
                                                ? { ...m, content: accumulatedContent }
                                                : m
                                            )
                                        };
                                    }
                                    return s;
                                }));
                                // scrollToBottom immediately inside the stream
                                chatEndRef.current?.scrollIntoView();
                            }
                        } catch (e) {
                            console.error("Parse error:", e);
                        }
                    }
                }
            }
        } catch (error) {
            console.error(error);
            setSessions(prev => prev.map(s => {
                if (s.id === sessionId) {
                    return {
                        ...s,
                        messages: s.messages.map(m => 
                            m.id === aiMessageId 
                            ? { ...m, content: "Error: Could not connect to API or request failed." }
                            : m
                        )
                    };
                }
                return s;
            }));
        } finally {
            setIsStreaming(false);
        }
    };

    return (
        <div className="layout-wrapper" style={{ background: "var(--bg)", color: "var(--text-main)" }}>
            {/* Mobile Header Menu */}
            <div className="hamburger" onClick={() => setSidebarOpen(true)}>
                <PanelLeft size={24} color="#fff" />
            </div>

            {/* Sidebar */}
            <div className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
                    <div className="brand" style={{ fontSize: '1.1rem', gap: '8px' }}>
                        <Brain size={20} color="var(--accent)" /> ResearchPilot
                    </div>
                    {sidebarOpen && (
                        <PanelLeftClose size={20} color="#fff" onClick={() => setSidebarOpen(false)} style={{ cursor: 'pointer' }} />
                    )}
                </div>

                <button className="new-chat-btn" onClick={handleNewChat}>
                    <Plus size={16} /> New Chat
                </button>

                <div style={{ flex: 1, overflowY: 'auto' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '12px', textTransform: 'uppercase' }}>Recent</div>
                    {sessions.map(s => (
                        <div 
                            key={s.id} 
                            className={`history-item ${activeSessionId === s.id ? 'active' : ''}`}
                            onClick={() => { setActiveSessionId(s.id); setSidebarOpen(false); }}
                        >
                            {s.title}
                        </div>
                    ))}
                </div>
            </div>

            {/* Overlay for mobile sidebar */}
            {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />}

            {/* Main Area */}
            <div className="main-chat-area">
                <div className="chat-container">
                    {currentMessages.length === 0 ? (
                        <div style={{ margin: 'auto', textAlign: 'center', opacity: 0.5, maxWidth: '400px' }}>
                            <Brain size={48} style={{ margin: '0 auto 16px auto', display: 'block' }} />
                            <h2>How can I help you with ML research?</h2>
                        </div>
                    ) : (
                        currentMessages.map((msg, i) => (
                            <div key={msg.id} className={msg.role === 'user' ? 'message-user' : 'message-ai'}>
                                {msg.role === 'user' ? (
                                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                                ) : (
                                    <div style={{ width: '100%' }}>
                                        {/* Name header for AI */}
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', fontSize: '0.85rem', color: 'var(--accent)', fontWeight: 600 }}>
                                            <Brain size={16} /> ResearchPilot {msg.model_used && <span style={{fontSize: '0.7rem', color: '#666', background: '#222', padding: '2px 6px', borderRadius: '4px'}}>{msg.model_used}</span>}
                                        </div>

                                        { /* Stream logic vs Final logic */
                                            isStreaming && i === currentMessages.length - 1 ? (
                                                <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                                                    {msg.content}
                                                    <span className="blinking-cursor"></span>
                                                </div>
                                            ) : (
                                                <>
                                                    <MessageRenderer content={msg.content} />
                                                    {/* Citations section if present */}
                                                    {msg.citations && msg.citations.length > 0 && (
                                                        <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
                                                            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '8px' }}>SOURCES</div>
                                                            <div className="citations-grid">
                                                                {msg.citations.map(c => (
                                                                    <div key={c.paper_id} className="citation-card">
                                                                        <div className="citation-meta">
                                                                            <span className="citation-id">{c.paper_id}</span>
                                                                        </div>
                                                                        <div className="citation-title">{c.title}</div>
                                                                        <a href={c.arxiv_url} target="_blank" rel="noopener noreferrer" className="citation-open" title="Open ArXiv PDF">
                                                                            <Search size={16} />
                                                                        </a>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                    {/* Feedback Row */}
                                                    {!isStreaming && msg.role === 'assistant' && msg.content && (
                                                        <FeedbackRow 
                                                            query={currentMessages[i-1]?.content || ""} 
                                                            time={msg.timing?.total_time_ms || 0} 
                                                            citationsCount={msg.citations?.length || 0}
                                                            model={msg.model_used || "unknown"}
                                                        />
                                                    )}
                                                </>
                                            )
                                        }
                                    </div>
                                )}
                            </div>
                        ))
                    )}
                    <div ref={chatEndRef} style={{ height: "1px" }} />
                </div>

                {/* Bottom Input Area */}
                <div className="bottom-input-bar">
                    <div className="bottom-input-bar-inner">
                        {/* Settings Popup inline */}
                        <AnimatePresence>
                            {settingsOpen && (
                                <motion.div 
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: 10 }}
                                    style={{ background: 'rgba(20,25,35,0.95)', border: '1px solid rgba(255,255,255,0.1)', padding: '12px', borderRadius: '12px', display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '8px' }}
                                >
                                    <select style={{ background: '#000', border: '1px solid #333', color: '#fff', padding: '6px 12px', borderRadius: '6px' }} value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
                                        <option value={3}>Top 3</option>
                                        <option value={5}>Top 5</option>
                                        <option value={10}>Top 10</option>
                                    </select>
                                    <select style={{ background: '#000', border: '1px solid #333', color: '#fff', padding: '6px 12px', borderRadius: '6px' }} value={category} onChange={(e) => setCategory(e.target.value)}>
                                        <option value="All">All Topics</option>
                                        <option value="cs.LG">cs.LG</option>
                                        <option value="cs.AI">cs.AI</option>
                                    </select>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        <div className="chat-input-wrapper">
                            <button onClick={() => setSettingsOpen(!settingsOpen)} style={{ color: settingsOpen ? 'var(--accent)' : 'var(--text-muted)' }}>
                                <Settings2 size={20} />
                            </button>
                            <textarea 
                                ref={textareaRef}
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && !e.shiftKey) {
                                        e.preventDefault();
                                        handleSend();
                                    }
                                }}
                                placeholder="Message ResearchPilot..."
                                className="chat-input"
                                rows={1}
                            />
                            <button onClick={handleSend} disabled={!query.trim() || isStreaming} className="send-btn">
                                <Send size={18} />
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div className="luminous-grid" />
        </div>
    );
}
