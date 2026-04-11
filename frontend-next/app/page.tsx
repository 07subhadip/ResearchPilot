'use client';

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
    Brain, Search, PanelLeftClose, PanelLeft, Plus,
    Send, Settings2, Trash2, Copy, Check, Star, ThumbsUp, ThumbsDown,
    Pin, Edit2, Check as CheckIcon
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
    pinned?: boolean;
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

const MessageRenderer = ({ content, isStreaming }: { content: string, isStreaming?: boolean }) => {
    const [displayed, setDisplayed] = useState(isStreaming ? "" : content);
    const [isThinking, setIsThinking] = useState(isStreaming && !content);

    useEffect(() => {
        if (!isStreaming) {
            setDisplayed(content);
            setIsThinking(false);
            return;
        }

        if (!content) {
            setIsThinking(true);
            return;
        }
        setIsThinking(false);

        if (displayed === content) return;

        const interval = setInterval(() => {
            setDisplayed(prev => {
                const diff = content.length - prev.length;
                if (diff <= 0) {
                    clearInterval(interval);
                    return content;
                }
                const charsToAdd = Math.max(1, Math.floor(diff / 5));
                return content.substring(0, prev.length + charsToAdd);
            });
        }, 15);

        return () => clearInterval(interval);
    }, [content, isStreaming, displayed]);

    if (isThinking) {
        return (
            <div className="thinking-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
            </div>
        );
    }

    // Match any Arxiv format number (YYYY.NNNNN) regardless of brackets or commas, and convert to Markdown link
    const processedContent = displayed.replace(/\b(\d{4}\.\d{4,5})\b/g, '[$1](CITATION:$1)');

    return (
        <div className="markdown-body">
            <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false }]]}
                urlTransform={(url) => url}
                components={{
                    code({node, inline, className, children, ...props}: any) {
                        const match = /language-(\w+)/.exec(className || '');
                        return !inline ? (
                            <CodeBlock language={match?.[1] || 'text'} code={String(children).replace(/\n$/, '')} />
                        ) : (
                            <code className="inline-code" {...props}>{children}</code>
                        )
                    },
                    a({node, href, children, ...props}: any) {
                        if (href?.startsWith('CITATION:')) {
                            return <CitationBadge id={href.replace('CITATION:', '')} />
                        }
                        return <a href={href} target="_blank" rel="noopener noreferrer" {...props}>{children}</a>
                    }
                }}
            >
                {processedContent + (isStreaming ? " █" : "")}
            </ReactMarkdown>
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
    const [apiStatus, setApiStatus] = useState<"connecting" | "online" | "offline">("connecting");

    const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
    const [editTitle, setEditTitle] = useState("");

    const handlePin = (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setSessions(prev => {
            const copy = [...prev];
            const idx = copy.findIndex(s => s.id === id);
            if (idx >= 0) copy[idx] = { ...copy[idx], pinned: !copy[idx].pinned };
            return copy.sort((a,b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0) || b.timestamp - a.timestamp);
        });
    };

    const handleDelete = (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setSessions(prev => prev.filter(s => s.id !== id));
        if (activeSessionId === id) setActiveSessionId(null);
    };

    const startEditing = (id: string, currentTitle: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setEditingSessionId(id);
        setEditTitle(currentTitle);
    };

    const saveTitle = (id: string, e: React.MouseEvent | React.KeyboardEvent) => {
        e.stopPropagation();
        if (!editTitle.trim()) { setEditingSessionId(null); return; }
        setSessions(prev => prev.map(s => s.id === id ? { ...s, title: editTitle } : s));
        setEditingSessionId(null);
    };

    useEffect(() => {
        const checkStatus = () => {
            fetch(`${API_URL}/health`)
                .then(res => setApiStatus(res.ok ? "online" : "offline"))
                .catch(() => setApiStatus("offline"));
        };
        checkStatus();
        const interval = setInterval(checkStatus, 30000);
        return () => clearInterval(interval);
    }, []);

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
                    filter_category: category === "All" ? undefined : category
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
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '32px' }}>
                    <div className="brand" style={{ fontSize: '1.2rem', gap: '10px' }}>
                        <div className="brand-icon"><Brain size={22} color="var(--accent)" /></div> ResearchPilot
                    </div>
                    {sidebarOpen && (
                        <PanelLeftClose size={20} color="#fff" onClick={() => setSidebarOpen(false)} style={{ cursor: 'pointer' }} />
                    )}
                </div>

                <motion.button 
                    whileHover={{ scale: 1.02, backgroundColor: "rgba(0, 240, 255, 0.1)", borderColor: "rgba(0, 240, 255, 0.3)" }}
                    whileTap={{ scale: 0.98 }}
                    className="new-chat-btn" 
                    onClick={handleNewChat}
                >
                    <Plus size={16} /> New Chat
                </motion.button>

                <div style={{ flex: 1, overflowY: 'auto' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '12px', textTransform: 'uppercase' }}>Recent</div>
                    {sessions.map(s => (
                        <div 
                            key={s.id} 
                            className={`history-item ${activeSessionId === s.id ? 'active' : ''}`}
                            onClick={() => { setActiveSessionId(s.id); setSidebarOpen(false); }}
                        >
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                                {editingSessionId === s.id ? (
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px', width: '100%' }}>
                                        <input 
                                            value={editTitle} 
                                            onChange={e => setEditTitle(e.target.value)}
                                            onClick={e => e.stopPropagation()}
                                            autoFocus
                                            onKeyDown={e => {
                                                if (e.key === 'Enter') saveTitle(s.id, e);
                                            }}
                                            style={{ background: 'rgba(0,0,0,0.5)', border: '1px solid var(--accent)', outline: 'none', color: '#fff', padding: '2px 6px', borderRadius: '4px', flex: 1, fontSize: '0.8rem' }}
                                        />
                                        <CheckIcon size={14} color="var(--success)" onClick={(e) => saveTitle(s.id, e)} style={{ cursor: 'pointer' }} />
                                    </div>
                                ) : (
                                    <>
                                        <div style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                            {s.pinned && <Pin size={12} color="var(--accent)" style={{ flexShrink: 0 }} />}
                                            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{s.title}</span>
                                        </div>
                                        <div className="history-actions">
                                            <Edit2 size={12} onClick={(e) => startEditing(s.id, s.title, e)} />
                                            <Pin size={14} onClick={(e) => handlePin(s.id, e)} className={s.pinned ? "active-pin" : ""} />
                                            <Trash2 size={12} onClick={(e) => handleDelete(s.id, e)} className="action-delete" />
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Overlay for mobile sidebar */}
            <AnimatePresence>
                {sidebarOpen && (
                    <motion.div 
                        initial={{ opacity: 0 }} 
                        animate={{ opacity: 1 }} 
                        exit={{ opacity: 0 }} 
                        className="sidebar-overlay" 
                        onClick={() => setSidebarOpen(false)} 
                    />
                )}
            </AnimatePresence>

            {/* Main Area */}
            <div className="main-chat-area">
                {/* Header API Status */}
                <div className="top-api-status">
                    <div className="nav-status">
                        <div className={`status-dot ${apiStatus === 'online' ? 'status-online' : 'status-offline'}`} />
                        {apiStatus === 'online' ? 'API Online' : apiStatus === 'connecting' ? 'Connecting...' : 'API Offline'}
                    </div>
                </div>

                <div className="chat-container">
                    {currentMessages.length === 0 ? (
                        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} style={{ margin: 'auto', textAlign: 'center', opacity: 0.8, maxWidth: '400px' }}>
                            <div className="hero-icon-container">
                                <Brain size={56} color="var(--accent)" />
                            </div>
                            <h2 style={{ fontSize: '1.6rem', fontWeight: 700, marginBottom: '12px' }}>Welcome to ResearchPilot</h2>
                            <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>Ask questions about Machine Learning research, get answers backed directly by ArXiv papers.</p>
                        </motion.div>
                    ) : (
                        currentMessages.map((msg, i) => (
                            <motion.div 
                                initial={{ opacity: 0, y: 15 }} 
                                animate={{ opacity: 1, y: 0 }} 
                                key={msg.id} 
                                className={msg.role === 'user' ? 'message-user' : 'message-ai'}
                            >
                                {msg.role === 'user' ? (
                                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                                ) : (
                                    <div style={{ width: '100%' }}>
                                        {/* Name header for AI */}
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', fontSize: '0.9rem', color: 'var(--accent)', fontWeight: 600 }}>
                                            <div className="ai-avatar"><Brain size={16} /></div> ResearchPilot {msg.model_used && <span className="model-badge">{msg.model_used}</span>}
                                        </div>

                                        { /* Stream logic vs Final logic */
                                            isStreaming && i === currentMessages.length - 1 ? (
                                                <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                                                    {msg.content}
                                                    <span className="blinking-cursor"></span>
                                                </div>
                                            ) : (
                                                <>
                                                    <MessageRenderer content={msg.content} isStreaming={isStreaming && i === currentMessages.length - 1} />
                                                    {/* Citations section if present */}
                                                    {msg.citations && msg.citations.length > 0 && (
                                                        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} style={{ marginTop: '24px', paddingTop: '20px', borderTop: '1px solid var(--border)' }}>
                                                            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '12px', letterSpacing: '0.05em' }}>SOURCES</div>
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
                                                        </motion.div>
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
                            </motion.div>
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
                            <motion.button 
                                whileHover={{ scale: 1.1, boxShadow: "0 0 20px rgba(0,240,255,0.6)", backgroundColor: "var(--accent)" }}
                                whileTap={{ scale: 0.9 }}
                                onClick={handleSend} 
                                disabled={!query.trim() || isStreaming} 
                                className="send-btn"
                            >
                                <Send size={18} />
                            </motion.button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div className="luminous-grid" />
        </div>
    );
}
