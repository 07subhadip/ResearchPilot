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
    Brain, Search, PanelLeftClose, PanelLeft, PanelLeftOpen, Plus,
    Send, Settings2, Trash2, Copy, Check, Star, ThumbsUp, ThumbsDown,
    Pin, Edit2, Check as CheckIcon, ArrowDown,
    Info, X, Server, Activity, Layers, Rocket
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
                const charsToAdd = Math.max(1, Math.floor(diff / 25));
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

    // Match any Arxiv format number (YYYY.NNNNN) regardless of brackets or commas, and convert to Markdown link. We eat the surrounding brackets to preserve design.
    let processedContent = displayed.replace(/\[?\s*\b(\d{4}\.\d{4,5})\b\s*\]?/g, '[$1](CITATION:$1)');
    
    // Force $$ block math onto separate lines so remarkMath parses it tightly as centered block math
    processedContent = processedContent.replace(/\$\$([\s\S]*?)\$\$/g, '\n\n$$\n$1\n$$\n\n');

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
                {processedContent}
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

// --- Custom Dropdown Component ---
const CustomDropdown = ({ label, options, value, onChange }: { label: string, options: {value: any, label: string}[], value: any, onChange: (val: any) => void }) => {
    const [isOpen, setIsOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    const selectedOption = options.find(opt => opt.value === value) || options[0];

    return (
        <div ref={containerRef} style={{ display: 'flex', flexDirection: 'column', gap: '4px', position: 'relative', minWidth: '160px' }}>
            <label style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 800, marginLeft: '4px', letterSpacing: '0.1em', textTransform: 'uppercase' }}>{label}</label>
            <div 
                onClick={() => setIsOpen(!isOpen)}
                style={{ 
                    background: 'rgba(255, 255, 255, 0.03)', 
                    border: '1px solid rgba(255, 255, 255, 0.1)', 
                    color: '#fff', 
                    padding: '10px 14px', 
                    borderRadius: '10px', 
                    cursor: 'pointer', 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    fontSize: '0.9rem',
                    fontWeight: 500,
                    transition: '0.2s',
                    userSelect: 'none'
                }}
                onMouseOver={e => e.currentTarget.style.borderColor = 'rgba(0, 240, 255, 0.4)'}
                onMouseOut={e => e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)'}
            >
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{selectedOption.label}</span>
                <motion.div animate={{ rotate: isOpen ? 180 : 0 }} style={{ display: 'flex', marginLeft: '20px' }}>
                    <ArrowDown size={14} opacity={0.5} />
                </motion.div>
            </div>

            <AnimatePresence>
                {isOpen && (
                    <motion.div 
                        initial={{ opacity: 0, y: -10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 5, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.95 }}
                        style={{ 
                            position: 'absolute', 
                            bottom: '100%', 
                            left: 0, 
                            right: 0, 
                            background: 'rgba(15, 20, 30, 0.98)', 
                            backdropFilter: 'blur(20px)',
                            border: '1px solid rgba(255, 255, 255, 0.15)', 
                            borderRadius: '12px', 
                            zIndex: 1000, 
                            padding: '6px',
                            boxShadow: '0 15px 40px rgba(0,0,0,0.6)',
                            marginBottom: '8px'
                        }}
                    >
                        {options.map((opt) => (
                            <div 
                                key={opt.value}
                                onClick={() => { onChange(opt.value); setIsOpen(false); }}
                                style={{ 
                                    padding: '10px 12px', 
                                    borderRadius: '8px', 
                                    cursor: 'pointer', 
                                    fontSize: '0.85rem',
                                    fontWeight: opt.value === value ? 600 : 400,
                                    color: opt.value === value ? 'var(--accent)' : '#fff',
                                    background: opt.value === value ? 'rgba(0, 240, 255, 0.05)' : 'transparent',
                                    transition: '0.2s'
                                }}
                                onMouseOver={e => { if(opt.value !== value) e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)'; }}
                                onMouseOut={e => { if(opt.value !== value) e.currentTarget.style.background = 'transparent'; }}
                            >
                                {opt.label}
                            </div>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
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
    const [filterYear, setFilterYear] = useState("All");
    const [apiStatus, setApiStatus] = useState<"connecting" | "online" | "offline">("connecting");
    const [desktopSidebarCollapsed, setDesktopSidebarCollapsed] = useState(false);
    const [showScrollDown, setShowScrollDown] = useState(false);
    const [showInfo, setShowInfo] = useState(false);
    const mainChatRef = useRef<HTMLDivElement>(null);

    const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
        const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
        const isNearBottom = scrollHeight - scrollTop - clientHeight < 150;
        setShowScrollDown(!isNearBottom);
    };


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

    const handleClearConversation = () => {
        if (!activeSessionId) return;
        setSessions(prev => prev.map(s => s.id === activeSessionId ? { ...s, messages: [] } : s));
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
            timestamp: Date.now() + 1,
            model_used: "Auto-Detecting..."
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

        const history = currentMessages
            .filter(m => m.role === "user" || m.role === "assistant")
            .map(m => ({
                role: m.role,
                content: m.content,
                citations: m.citations || []
            }))
            .slice(-20);

        try {
            const res = await fetch(`${API_URL}/query/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    question: originalQuery, 
                    history: history,
                    top_k: topK, 
                    filter_category: category === "All" ? undefined : category,
                    filter_year_gte: filterYear === "All" ? undefined : parseInt(filterYear, 10)
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
            <div className={`sidebar ${sidebarOpen ? 'open' : ''} ${desktopSidebarCollapsed ? 'collapsed' : ''}`}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '32px', whiteSpace: 'nowrap' }}>
                    <div className="brand" style={{ fontSize: '1.2rem', gap: '10px' }}>
                        <div className="brand-icon"><Brain size={22} color="var(--accent)" /></div> ResearchPilot
                    </div>
                    <div onClick={() => { setSidebarOpen(false); setDesktopSidebarCollapsed(true); }} style={{ cursor: 'pointer', opacity: 0.7 }}>
                        <PanelLeftClose size={20} color="#fff" />
                    </div>
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

                <div className="sidebar-footer">
                    <a href="https://github.com/07subhadip" target="_blank" rel="noopener noreferrer" className="github-link" style={{ display: 'flex', alignItems: 'center', gap: '10px', color: 'var(--accent)', textDecoration: 'none', fontSize: '0.9rem', padding: '12px', borderRadius: '8px', transition: '0.3s', background: 'rgba(0, 240, 255, 0.1)', border: '1px solid rgba(0, 240, 255, 0.3)', boxShadow: '0 0 15px rgba(0, 240, 255, 0.2)' }} onMouseOver={e => e.currentTarget.style.boxShadow = '0 0 25px rgba(0, 240, 255, 0.6)'} onMouseOut={e => e.currentTarget.style.boxShadow = '0 0 15px rgba(0, 240, 255, 0.2)'}>
                        <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-github"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg> 
                        <span style={{ fontWeight: 600, letterSpacing: '0.05em' }}>@07subhadip</span>
                    </a>
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
            <div className="desktop-toggle-sidebar" onClick={() => setDesktopSidebarCollapsed(!desktopSidebarCollapsed)}>
                {desktopSidebarCollapsed ? <PanelLeftOpen size={20} /> : <PanelLeftClose size={20} />}
            </div>

            <div className="main-chat-area" onScroll={handleScroll} ref={mainChatRef}>
                <AnimatePresence>
                    {showScrollDown && (
                        <motion.button 
                            initial={{ opacity: 0, scale: 0.8 }} 
                            animate={{ opacity: 1, scale: 1 }} 
                            exit={{ opacity: 0, scale: 0.8 }} 
                            className="scroll-down-btn" 
                            onClick={scrollToBottom}
                        >
                            <ArrowDown size={20} />
                        </motion.button>
                    )}
                </AnimatePresence>
                <div className="top-api-status" style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                    <button onClick={() => setShowInfo(true)} className="nav-icon-btn" aria-label="Project Info" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', padding: '6px', borderRadius: '50%', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Info size={16} />
                    </button>
                    {activeSessionId && currentMessages.length > 0 && (
                        <button onClick={handleClearConversation} className="nav-icon-btn" aria-label="Clear Conversation" title="Clear current conversation context" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', padding: '6px 12px', borderRadius: '16px', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', fontSize: '0.8rem' }}>
                            <Trash2 size={14} /> Clear context
                        </button>
                    )}
                    <div className="nav-status">
                        <div className={`status-dot ${apiStatus === 'online' ? 'status-online' : 'status-offline'}`} />
                        {apiStatus === 'online' ? 'API Online' : apiStatus === 'connecting' ? 'Connecting...' : 'API Offline'}
                    </div>
                </div>

                <AnimatePresence>
                    {showInfo && (
                        <div className="info-modal-backdrop" onClick={() => setShowInfo(false)} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(8px)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95, y: 30 }}
                                animate={{ opacity: 1, scale: 1, y: 0 }}
                                exit={{ opacity: 0, scale: 0.95, y: 30 }}
                                onClick={(e) => e.stopPropagation()}
                                className="cyber-panel info-modal"
                                style={{ 
                                    background: 'linear-gradient(135deg, rgba(20, 25, 40, 0.95), rgba(10, 15, 25, 0.98))', 
                                    border: '1px solid rgba(0, 240, 255, 0.3)', 
                                    padding: '40px', 
                                    borderRadius: '24px', 
                                    maxWidth: '650px', 
                                    width: '90%', 
                                    position: 'relative', 
                                    maxHeight: '85vh', 
                                    overflowY: 'auto', 
                                    boxShadow: '0 20px 50px rgba(0, 0, 0, 0.9), 0 0 40px rgba(0, 240, 255, 0.1)' 
                                }}
                            >
                                <button className="modal-close" onClick={() => setShowInfo(false)} style={{ position: 'absolute', top: '16px', right: '16px', background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
                                    <X size={18} />
                                </button>
                                <h2 style={{ margin: '0 0 16px 0' }}>ResearchPilot Console</h2>
                                <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "16px" }}>
                                    <div style={{ background: "rgba(138, 43, 226, 0.15)", border: "1px solid rgba(138, 43, 226, 0.4)", padding: "6px 14px", borderRadius: "99px", fontSize: "0.75rem", color: "var(--accent-2)", fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase" }}>
                                        Lead Architect
                                    </div>
                                    <span style={{ fontFamily: "'Dancing Script', cursive", fontSize: "1.8rem", fontWeight: 700, background: "linear-gradient(135deg, #fff 20%, var(--accent-2) 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", letterSpacing: "0.05em", transform: "translateY(-2px)" }}>Subhadip Hensh</span>
                                </div>
                                <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: '24px 0' }} />
                                <h3><Server size={18} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '8px' }} /> System Overview</h3>
                                <p style={{ fontSize: "0.95rem", lineHeight: 1.7, marginBottom: "16px", color: 'var(--text-muted)' }}>ResearchPilot is a high-performance RAG engine tailored for Machine Learning literature. It features hybrid sparse-dense searching, advanced cross-encoder reranking, and GPU-driven vector indexing via Qdrant.</p>
                                <h3><Activity size={18} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '8px' }} /> Current Operational Capacity</h3>
                                <ul style={{ fontSize: "0.95rem", lineHeight: 1.7, color: 'var(--text-muted)', paddingLeft: '20px' }}>
                                    <li style={{ marginBottom: '12px' }}><strong>Current Index</strong> Synthesizing 300k+ semantic chunks isolated from 3,500+ premium AI & ML research papers.</li>
                                    <li><strong>Data Categories</strong> Comprehensive coverage across specialized domains:
                                        <ul style={{ paddingLeft: '20px', marginTop: '8px', fontSize: '0.9rem' }}>
                                            <li><strong>cs.LG</strong>: Machine Learning</li>
                                            <li><strong>cs.AI</strong>: Artificial Intelligence</li>
                                            <li><strong>stat.ML</strong>: Machine Learning (Statistics)</li>
                                            <li><strong>cs.CL</strong>: Computation and Language (NLP)</li>
                                            <li><strong>cs.CV</strong>: Computer Vision & Pattern Recognition</li>
                                            <li><strong>cs.RO</strong>: Robotics</li>
                                        </ul>
                                    </li>
                                </ul>
                                <h3><Layers size={18} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '8px' }} /> Core Technology Stack</h3>
                                <ul style={{ fontSize: "0.95rem", lineHeight: 1.7, color: 'var(--text-muted)', paddingLeft: '20px' }}>
                                    <li style={{ marginBottom: '8px' }}><strong>Frontend Application</strong> Next.js 16 (App Router), React, Framer Motion, Vanilla CSS.</li>
                                    <li style={{ marginBottom: '8px' }}><strong>Backend Environment</strong> Python, FastAPI, Uvicorn, Pydantic.</li>
                                    <li style={{ marginBottom: '8px' }}><strong>Vector Database Engine</strong> Qdrant (GPU Accelerated Dense Vectors).</li>
                                    <li style={{ marginBottom: '8px' }}><strong>RAG Processing Pipeline</strong> SentenceTransformers (BGE-base-en-v1.5), BM25 Sparse Search, Cross-Encoder Reranking.</li>
                                    <li style={{ marginBottom: '8px' }}><strong>Multi-Modal LLM Fabric</strong> Dynamic routing between Qwen 2.5 72B (Primary), LLaMA 3.3 70B (Fallback), and Qwen 2.5 Coder 7B (Code).</li>
                                    <li><strong>Mathematics Engine</strong> KaTeX & React-Markdown for fully dynamic native LaTeX equations.</li>
                                </ul>
                                <h3><Rocket size={18} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '8px' }} /> Phase 2: In-Progress Architecture</h3>
                                <ul style={{ fontSize: "0.95rem", lineHeight: 1.7, color: 'var(--text-muted)', paddingLeft: '20px' }}>
                                    <li style={{ marginBottom: '8px' }}><strong>Massive Data Expansion</strong> Scaling dataset soon to 10,000+ — 20,000+ ML papers spanning NLP, Computer Vision, and Robotics.</li>
                                    <li style={{ marginBottom: '8px' }}><strong>Distributed Hardware Execution</strong> Scaling ingestion logic to cloud-based GPU clusters for extreme speed.</li>
                                    <li><strong>Multi-modal Analysis</strong> Soon integrating visual graph and chart processing abilities into the synthesis engine.</li>
                                </ul>
                            </motion.div>
                        </div>
                    )}
                </AnimatePresence>

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
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', fontSize: '0.85rem', color: 'var(--accent)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                            <div className="ai-avatar" style={{ background: 'rgba(0, 240, 255, 0.1)', border: '1px solid rgba(0, 240, 255, 0.3)', padding: '6px', borderRadius: '8px' }}><Brain size={14} /></div> 
                                            ResearchPilot 
                                            <span className="model-badge" style={{ background: 'rgba(255,255,255,0.05)', padding: '2px 8px', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                                                {msg.model_used || "Auto-Detecting..."}
                                            </span>
                                            {i >= 2 && (
                                                <span style={{ fontSize: '0.7rem', background: 'rgba(138, 43, 226, 0.15)', border: '1px solid rgba(138, 43, 226, 0.3)', padding: '2px 8px', borderRadius: '4px', color: 'var(--accent-2)', marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                    <Layers size={10} /> Using conversation context
                                                </span>
                                            )}
                                        </div>

                                        <>
                                            <MessageRenderer content={msg.content} isStreaming={isStreaming && i === currentMessages.length - 1} />
                                            {/* Citations section if present */}
                                            {(!isStreaming || i !== currentMessages.length - 1) && msg.citations && msg.citations.length > 0 && (
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
                                            {(!isStreaming || i !== currentMessages.length - 1) && msg.role === 'assistant' && msg.content && (
                                                <FeedbackRow 
                                                    query={currentMessages[i-1]?.content || ""} 
                                                    time={msg.timing?.total_time_ms || 0} 
                                                    citationsCount={msg.citations?.length || 0}
                                                    model={msg.model_used || "unknown"}
                                                />
                                            )}
                                        </>
                                    </div>
                                )}
                            </motion.div>
                        ))
                    )}
                    <div ref={chatEndRef} style={{ height: "1px" }} />
                </div>

                {/* Bottom Input Area */}
                <div className="bottom-input-bar" style={{ 
                    left: desktopSidebarCollapsed ? '0' : '260px', 
                    transition: 'left 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    width: 'auto'
                }}>
                    <div className="bottom-input-bar-inner">
                        {/* Settings Popup inline */}
                        <AnimatePresence>
                            {settingsOpen && (
                                <motion.div 
                                    initial={{ opacity: 0, y: 15, scale: 0.95 }}
                                    animate={{ opacity: 1, y: 0, scale: 1 }}
                                    exit={{ opacity: 0, y: 15, scale: 0.95 }}
                                    style={{ 
                                        background: 'rgba(15, 20, 30, 0.95)', 
                                        backdropFilter: 'blur(20px)',
                                        border: '1px solid rgba(0, 240, 255, 0.2)', 
                                        padding: '16px', 
                                        borderRadius: '16px', 
                                        display: 'flex', 
                                        gap: '16px', 
                                        justifyContent: 'space-between',
                                        flexWrap: 'wrap', 
                                        marginBottom: '12px',
                                        boxShadow: '0 10px 40px rgba(0,0,0,0.5)'
                                    }}
                                >
                                    <div style={{ flex: 1, minWidth: '160px' }}>
                                        <CustomDropdown 
                                            label="RETRIEVAL DEPTH"
                                            value={topK}
                                            onChange={setTopK}
                                            options={[
                                                { value: 3, label: 'Fast (3 Papers)' },
                                                { value: 5, label: 'Balanced (5 Papers)' },
                                                { value: 10, label: 'Deep (10 Papers)' }
                                            ]}
                                        />
                                    </div>

                                    <div style={{ flex: 1, minWidth: '160px' }}>
                                        <CustomDropdown 
                                            label="RESEARCH DOMAIN"
                                            value={category}
                                            onChange={setCategory}
                                            options={[
                                                { value: 'All', label: 'Global Search' },
                                                { value: 'cs.LG', label: 'cs.LG (Machine Learning)' },
                                                { value: 'cs.AI', label: 'cs.AI (Artificial Intelligence)' },
                                                { value: 'stat.ML', label: 'stat.ML (ML Stats)' },
                                                { value: 'cs.CL', label: 'cs.CL (NLP/Language)' },
                                                { value: 'cs.CV', label: 'cs.CV (Vision)' },
                                                { value: 'cs.RO', label: 'cs.RO (Robotics)' }
                                            ]}
                                        />
                                    </div>

                                    <div style={{ flex: 1, minWidth: '160px' }}>
                                        <CustomDropdown 
                                            label="RECENCY FILTER"
                                            value={filterYear}
                                            onChange={setFilterYear}
                                            options={[
                                                { value: 'All', label: 'Legacy & Modern' },
                                                { value: '2024', label: '2024 (Latest)' },
                                                { value: '2023', label: '2023+' },
                                                { value: '2022', label: '2022+' },
                                                { value: '2021', label: '2021+' },
                                                { value: '2020', label: '2020+' }
                                            ]}
                                        />
                                    </div>
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
