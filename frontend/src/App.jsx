import { useState, useEffect } from 'react';

// Generate a random, persistent student session ID for our Upstash Redis layer
const SESSION_ID = "student_session_" + Math.random().toString(36).substring(2, 8);

async function loadChatHistory(sessionId) {
  const response = await fetch(`http://127.0.0.1:8000/api/session/${sessionId}`);
  const data = await response.json();
  return data.history || [];
}

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [history, setHistory] = useState([]);
  const [aiResponse, setAiResponse] = useState("");
  const [activeTab, setActiveTab] = useState('search'); 

  // Fetch conversation timeline from Upstash Redis via our API
  const fetchChatHistory = async () => {
    try {
      setHistory(await loadChatHistory(SESSION_ID));
    } catch (err) {
      console.error("Failed to fetch session memory blueprint:", err);
    }
  };

  // Handle live vector query execution
  const handleSearch = async (e, customQuery = null) => {
    if (e) e.preventDefault();
    const searchQuery = customQuery || query;
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: SESSION_ID,
          query: searchQuery,
          limit: 3,
          min_trust_score: 0.1
        })
      });
      const data = await response.json();
      setResults(data.results || []);
      setAiResponse(data.ai_response || "");
      
      // Refresh memory log tracking tab after each query.
      fetchChatHistory();
    } catch (err) {
      console.error("Search API Error:", err);
      alert("Backend API is offline! Make sure your Uvicorn server is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let ignore = false;

    loadChatHistory(SESSION_ID)
      .then((nextHistory) => {
        if (!ignore) {
          setHistory(nextHistory);
        }
      })
      .catch((err) => {
        console.error("Failed to fetch session memory blueprint:", err);
      });

    return () => {
      ignore = true;
    };
  }, []);

  return (
    <div style={styles.container}>
      {/* Top Navigation / Header */}
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>Agentic Career Intelligence</h1>
          <p style={styles.subtitle}>Vector Analytics Engine & Dynamic Source Trust Matrix</p>
        </div>
        <div style={styles.sessionBadge}>
          <span style={styles.statusDot}></span>
          Session: <strong style={{color: '#61dafb', marginLeft: '5px'}}>{SESSION_ID}</strong>
        </div>
      </header>

      {/* Internal Navigation Tabs */}
      <div style={styles.tabContainer}>
        <button 
          onClick={() => setActiveTab('search')} 
          style={{...styles.tabButton, ...(activeTab === 'search' ? styles.activeTab : {})}}
        >
          Semantic Database Search
        </button>
        <button 
          onClick={() => { setActiveTab('history'); fetchChatHistory(); }} 
          style={{...styles.tabButton, ...(activeTab === 'history' ? styles.activeTab : {})}}
        >
          Cloud Memory Governance (Upstash)
        </button>
      </div>

      {/* Main Content Pane */}
      <main style={styles.mainContent}>
        {activeTab === 'search' ? (
          <div>
            {/* Quick Clicks / Recommendation Badges */}
            <div style={styles.suggestions}>
              <span style={styles.suggestionLabel}>Quick Filters:</span>
              {["Python Backend Engineer requirements", "Fiserv tested data analyst parameters", "Cloud DevOps Engineer tools"].map((text, idx) => (
                <button key={idx} onClick={(e) => { setQuery(text); handleSearch(e, text); }} style={styles.badge}>
                  {text}
                </button>
              ))}
            </div>

            {/* Main Ingestion Input Form */}
            <form onSubmit={handleSearch} style={styles.form}>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask about engineering roles, tool stacks, or target placement metrics..."
                style={styles.input}
              />
              <button type="submit" disabled={loading} style={styles.submitButton}>
                {loading ? 'Processing...' : 'Search Vector Space'}
              </button>
            </form>

            {/* Live Agentic GraphRAG Insights Display Box */}
            {aiResponse && (
              <div style={styles.aiInsightBox}>
                <h3 style={styles.aiHeader}>
                  <span>AI</span> Live Agentic GraphRAG Insights Analysis
                </h3>
                <p style={styles.aiText}>
                  {aiResponse}
                </p>
              </div>
            )}

            {/* Results Grid Display */}
            <div style={styles.resultsGrid}>
              {results.length > 0 ? (
                results.map((item, idx) => (
                  <div key={idx} style={styles.card}>
                    <div style={styles.cardHeader}>
                      <h3 style={styles.cardTitle}>{item.title}</h3>
                      <span style={{
                        ...styles.trustBadge, 
                        backgroundColor: item.match_score > 0.7 ? '#132d21' : '#2a2433',
                        color: item.match_score > 0.7 ? '#4caf50' : '#d8b4fe',
                        borderColor: item.match_score > 0.7 ? '#4caf50' : '#a855f7'
                      }}>
                        {item.confidence_label || 'Match'}: {item.match_score ?? item.trust_score}
                      </span>
                    </div>
                    <div style={styles.scoreGrid}>
                      <span>Semantic Match: <strong>{item.semantic_score ?? 'n/a'}</strong></span>
                      <span>Source Trust: <strong>{item.source_trust ?? 'n/a'}</strong></span>
                      <span>Entity: <strong>{item.type}</strong></span>
                    </div>
                    <p style={styles.cardDesc}>{item.context_description}</p>
                    {item.match_reasons?.length > 0 && (
                      <div style={styles.reasonRow}>
                        {item.match_reasons.map((reason, reasonIdx) => (
                          <span key={reasonIdx} style={styles.reasonChip}>{reason}</span>
                        ))}
                      </div>
                    )}
                    {item.skills?.length > 0 && (
                      <div style={styles.detailBlock}>
                        <strong>Priority Skills:</strong> {item.skills.slice(0, 8).join(', ')}
                      </div>
                    )}
                    {item.tools?.length > 0 && (
                      <div style={styles.detailBlock}>
                        <strong>Tools:</strong> {item.tools.slice(0, 8).join(', ')}
                      </div>
                    )}
                  </div>
                ))
              ) : (
                !loading && <div style={styles.emptyState}>No vectors retrieved yet. Input a parameter query above to trace insights live.</div>
              )}
            </div>
          </div>
        ) : (
          /* Cloud Session Memory Log Layout */
          <div style={styles.historyContainer}>
            <h2 style={{color: '#ffffff', marginBottom: '15px', fontSize: '1.2rem'}}>Active Upstash Redis JSON Pipeline Logs</h2>
            {history.length > 0 ? (
              history.map((msg, idx) => (
                <div key={idx} style={{
                  ...styles.chatBubble, 
                  backgroundColor: msg.role === 'user' ? '#1f293d' : '#1a1f2c',
                  borderLeft: msg.role === 'user' ? '4px solid #61dafb' : '4px solid #a855f7'
                }}>
                  <strong style={{color: msg.role === 'user' ? '#61dafb' : '#a855f7', display: 'block', marginBottom: '5px'}}>
                    {msg.role.toUpperCase()}
                  </strong>
                  <p style={{margin: 0, color: '#e2e8f0', lineHeight: '1.5'}}>{msg.content}</p>
                </div>
              ))
            ) : (
              <div style={styles.emptyState}>No state timeline found in cloud memory. Execute queries to build interaction chains.</div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

// Formal High-Tech CSS-in-JS Styles
const styles = {
  container: { backgroundColor: '#0f141c', minHeight: '100vh', color: '#abb2bf', fontFamily: 'system-ui, -apple-system, sans-serif' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '20px 40px', borderBottom: '1px solid #1e2530', backgroundColor: '#141b24' },
  title: { margin: 0, fontSize: '1.6rem', color: '#ffffff', fontWeight: '600' },
  subtitle: { margin: '4px 0 0 0', fontSize: '0.85rem', color: '#636d7e' },
  sessionBadge: { display: 'flex', alignItems: 'center', backgroundColor: '#1e2530', padding: '8px 14px', borderRadius: '6px', fontSize: '0.85rem', color: '#abb2bf' },
  statusDot: { width: '8px', height: '8px', backgroundColor: '#4caf50', borderRadius: '50%', marginRight: '8px', display: 'inline-block' },
  tabContainer: { display: 'flex', gap: '10px', padding: '20px 40px 0 40px' },
  tabButton: { backgroundColor: 'transparent', border: 'none', color: '#636d7e', padding: '10px 16px', cursor: 'pointer', fontSize: '0.95rem', fontWeight: '500', transition: '0.2s', borderBottom: '2px solid transparent' },
  activeTab: { color: '#ffffff', borderBottom: '2px solid #61dafb' },
  mainContent: { padding: '20px 40px' },
  suggestions: { display: 'flex', flexWrap: 'wrap', gap: '10px', alignItems: 'center', marginBottom: '20px' },
  suggestionLabel: { fontSize: '0.85rem', color: '#636d7e', fontWeight: '500' },
  badge: { backgroundColor: '#141b24', border: '1px solid #1e2530', color: '#61dafb', padding: '6px 12px', borderRadius: '20px', cursor: 'pointer', fontSize: '0.8rem', transition: '0.2s' },
  form: { display: 'flex', gap: '15px', marginBottom: '30px' },
  input: { flex: 1, backgroundColor: '#141b24', border: '1px solid #1e2530', borderRadius: '8px', padding: '14px 20px', color: '#ffffff', fontSize: '1rem', outline: 'none' },
  submitButton: { backgroundColor: '#61dafb', color: '#0f141c', border: 'none', borderRadius: '8px', padding: '0 24px', fontSize: '0.95rem', fontWeight: '600', cursor: 'pointer', transition: '0.2s' },
  resultsGrid: { display: 'flex', flexDirection: 'column', gap: '15px' },
  card: { backgroundColor: '#141b24', border: '1px solid #1e2530', borderRadius: '10px', padding: '20px', transition: '0.2s' },
  cardHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' },
  cardTitle: { margin: 0, fontSize: '1.2rem', color: '#ffffff' },
  trustBadge: { fontSize: '0.75rem', fontWeight: '600', padding: '4px 10px', borderRadius: '4px', border: '1px solid' },
  cardType: { margin: '0 0 10px 0', fontSize: '0.8rem', color: '#636d7e' },
  cardDesc: { margin: 0, fontSize: '0.95rem', color: '#abb2bf', lineHeight: '1.6' },
  scoreGrid: { display: 'flex', flexWrap: 'wrap', gap: '8px 18px', color: '#7d8796', fontSize: '0.8rem', marginBottom: '10px' },
  reasonRow: { display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '14px' },
  reasonChip: { backgroundColor: '#1f293d', color: '#c4d8ff', border: '1px solid #2d3a52', borderRadius: '4px', padding: '4px 8px', fontSize: '0.75rem' },
  detailBlock: { color: '#9da8b8', fontSize: '0.82rem', lineHeight: '1.5', marginTop: '10px' },
  emptyState: { textAlign: 'center', padding: '60px', color: '#636d7e', border: '1px dashed #1e2530', borderRadius: '10px', fontSize: '0.95rem' },
  historyContainer: { display: 'flex', flexDirection: 'column', gap: '12px' },
  chatBubble: { padding: '15px 20px', borderRadius: '8px', border: '1px solid #1e2530' },
  
  // High-Tech Dark Style Additions for the AI Container Layer
  aiInsightBox: { backgroundColor: '#111827', border: '1px solid #a855f7', borderRadius: '10px', padding: '22px', marginBottom: '25px', boxShadow: '0 4px 20px rgba(168, 85, 247, 0.15)' },
  aiHeader: { color: '#a855f7', marginTop: 0, fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '10px', borderBottom: '1px solid #1f293d', paddingBottom: '10px' },
  aiText: { color: '#e2e8f0', lineHeight: '1.65', fontSize: '0.95rem', whiteSpace: 'pre-line', margin: '12px 0 0 0' }
};

export default App;
