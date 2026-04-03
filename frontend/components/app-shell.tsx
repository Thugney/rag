"use client";

import {
  ChangeEvent,
  FormEvent,
  KeyboardEvent,
  startTransition,
  useDeferredValue,
  useEffect,
  useRef,
  useState
} from "react";
import {
  Bot,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  Database,
  FileStack,
  FolderOpen,
  LoaderCircle,
  LockKeyhole,
  Mail,
  Menu,
  Paperclip,
  Plus,
  Search,
  Send,
  ShieldCheck,
  UserRound,
  X
} from "lucide-react";

import {
  ChatResponse,
  DocumentRecord,
  HealthResponse,
  IndexJobResponse,
  LLMValidationResponse,
  MessageRecord,
  SessionSummary,
  SettingsResponse,
  createSession,
  getDocuments,
  getHealth,
  getIndexJob,
  getSessionMessages,
  getSessions,
  getSettings,
  sendChatQuery,
  startIndexJob,
  uploadDocuments,
  validateLLM
} from "../lib/api";

type StatusTone = "neutral" | "success" | "error";
type PanelKey = "documents" | "sources" | "settings" | null;

const PROFILE_STORAGE_KEY = "ragagument-profile";

function formatDateTime(value: string) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit"
  }).format(parsed);
}

function formatFileSize(sizeBytes: number) {
  if (sizeBytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(sizeBytes / 1024))} KB`;
  }

  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
}

function documentStatusLabel(document: DocumentRecord) {
  switch (document.processing_status) {
    case "indexed":
      return "Indexed";
    case "indexing":
      return "Indexing";
    case "queued":
      return "Queued";
    case "failed":
      return "Needs review";
    default:
      return "Uploaded";
  }
}

function buildJobMessage(job: IndexJobResponse) {
  if (job.status === "completed") {
    return `Indexed ${job.indexed_documents} document${job.indexed_documents === 1 ? "" : "s"} into ${job.indexed_chunks} chunks.`;
  }

  if (job.status === "failed") {
    return job.error ?? "Background indexing failed.";
  }

  return `Index job ${job.status}. ${job.document_ids.length} document${job.document_ids.length === 1 ? "" : "s"} in queue.`;
}

export function AppShell() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [llmValidation, setLlmValidation] = useState<LLMValidationResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [messages, setMessages] = useState<MessageRecord[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [composer, setComposer] = useState("");
  const [sources, setSources] = useState<ChatResponse["sources"]>([]);
  const [status, setStatus] = useState("Connecting to the enterprise workspace.");
  const [statusTone, setStatusTone] = useState<StatusTone>("neutral");
  const [chatBusy, setChatBusy] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [activeJob, setActiveJob] = useState<IndexJobResponse | null>(null);
  const [activePanel, setActivePanel] = useState<PanelKey>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [desktopRailOpen, setDesktopRailOpen] = useState(true);
  const [composerMenuOpen, setComposerMenuOpen] = useState(false);
  const [historyExpanded, setHistoryExpanded] = useState(true);
  const [sessionSearch, setSessionSearch] = useState("");
  const [profileName, setProfileName] = useState("Workspace Admin");
  const [userEmail, setUserEmail] = useState("admin@ragagument.local");
  const [jurisdiction, setJurisdiction] = useState("EU / EEA");
  const [retentionPolicy, setRetentionPolicy] = useState("30 days");
  const [complianceMode, setComplianceMode] = useState("GDPR strict");
  const composerSurfaceRef = useRef<HTMLDivElement | null>(null);

  const deferredSessionSearch = useDeferredValue(sessionSearch);
  const filteredSessions = sessions.filter((session) => {
    const query = deferredSessionSearch.trim().toLowerCase();
    if (!query) {
      return true;
    }

    return (
      session.title.toLowerCase().includes(query) ||
      session.start_time.toLowerCase().includes(query)
    );
  });

  const indexedDocuments = documents.filter((document) => document.processing_status === "indexed");
  const pendingDocuments = documents.filter((document) => document.processing_status !== "indexed");
  const isEmptySession = messages.length === 0;
  const jobIsRunning = Boolean(activeJob && ["queued", "running"].includes(activeJob.status));
  const composerStatus = jobIsRunning
    ? `Indexing ${activeJob?.document_ids.length ?? 0} doc${activeJob?.document_ids.length === 1 ? "" : "s"}`
    : pendingDocuments.length > 0
      ? `${pendingDocuments.length} pending`
      : documents.length > 0
        ? "Index ready"
        : "Awaiting upload";
  const userInitials =
    profileName
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase() ?? "")
      .join("") || "U";

  useEffect(() => {
    void loadWorkspace();
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const stored = window.localStorage.getItem(PROFILE_STORAGE_KEY);
    if (!stored) {
      return;
    }

    try {
      const parsed = JSON.parse(stored) as {
        profileName?: string;
        userEmail?: string;
        jurisdiction?: string;
        retentionPolicy?: string;
        complianceMode?: string;
      };

      if (parsed.profileName) {
        setProfileName(parsed.profileName);
      }
      if (parsed.userEmail) {
        setUserEmail(parsed.userEmail);
      }
      if (parsed.jurisdiction) {
        setJurisdiction(parsed.jurisdiction);
      }
      if (parsed.retentionPolicy) {
        setRetentionPolicy(parsed.retentionPolicy);
      }
      if (parsed.complianceMode) {
        setComplianceMode(parsed.complianceMode);
      }
    } catch {
      window.localStorage.removeItem(PROFILE_STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    window.localStorage.setItem(
      PROFILE_STORAGE_KEY,
        JSON.stringify({
          profileName,
          userEmail,
          jurisdiction,
          retentionPolicy,
          complianceMode
        })
    );
  }, [complianceMode, jurisdiction, profileName, retentionPolicy, userEmail]);

  useEffect(() => {
    if (!activeJob || !["queued", "running"].includes(activeJob.status)) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void pollIndexJob(activeJob.job_id);
    }, 2000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [activeJob]);

  useEffect(() => {
    if (!composerMenuOpen) {
      return;
    }

    function handlePointerDown(event: PointerEvent) {
      if (!composerSurfaceRef.current?.contains(event.target as Node)) {
        setComposerMenuOpen(false);
      }
    }

    document.addEventListener("pointerdown", handlePointerDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
    };
  }, [composerMenuOpen]);

  function setWorkspaceStatus(message: string, tone: StatusTone = "neutral") {
    setStatus(message);
    setStatusTone(tone);
  }

  async function loadWorkspace(options?: { preserveStatus?: boolean }) {
    try {
      const [healthData, settingsData, llmData, documentsData, sessionsData] = await Promise.all([
        getHealth(),
        getSettings(),
        validateLLM(),
        getDocuments(),
        getSessions()
      ]);

      const runningJobId = documentsData.find((document) => document.active_index_job_id)?.active_index_job_id;
      let currentJob: IndexJobResponse | null = null;
      if (runningJobId) {
        currentJob = await getIndexJob(runningJobId);
      }

      startTransition(() => {
        setHealth(healthData);
        setSettings(settingsData);
        setLlmValidation(llmData);
        setDocuments(documentsData);
        setSessions(sessionsData);
        setActiveJob(currentJob);
      });

      if (!options?.preserveStatus) {
        setWorkspaceStatus(
          llmData.valid
            ? "Workspace ready. Uploads auto-index in the background."
            : llmData.message,
          llmData.valid ? "success" : "error"
        );
      }
    } catch {
      setWorkspaceStatus("API not reachable. Start the FastAPI service to continue.", "error");
    }
  }

  async function pollIndexJob(jobId: string) {
    try {
      const job = await getIndexJob(jobId);
      setActiveJob(job);

      if (job.status === "completed") {
        await loadWorkspace({ preserveStatus: true });
        setWorkspaceStatus(buildJobMessage(job), "success");
      } else if (job.status === "failed") {
        await loadWorkspace({ preserveStatus: true });
        setWorkspaceStatus(buildJobMessage(job), "error");
      }
    } catch {
      setWorkspaceStatus("Could not refresh indexing status.", "error");
    }
  }

  async function openSession(sessionId: string) {
    try {
      const history = await getSessionMessages(sessionId);
      startTransition(() => {
        setActiveSessionId(sessionId);
        setMessages(history);
        setSources([]);
      });
      setSidebarOpen(false);
      setWorkspaceStatus("Conversation history loaded.", "neutral");
    } catch {
      setWorkspaceStatus("Could not load session history.", "error");
    }
  }

  function handleNewChat() {
    startTransition(() => {
      setActiveSessionId(null);
      setMessages([]);
      setSources([]);
    });
    setActivePanel(null);
    setSidebarOpen(false);
    setWorkspaceStatus("New conversation ready. Uploads continue indexing in the background.", "neutral");
  }

  function togglePanel(panel: Exclude<PanelKey, null>) {
    setActivePanel((current) => (current === panel ? null : panel));
    setComposerMenuOpen(false);
  }

  function closePanel() {
    setActivePanel(null);
  }

  function handleSettingsAction(actionLabel: string) {
    setActivePanel("settings");
    setWorkspaceStatus(`${actionLabel} will be wired to the auth backend in the next SaaS phase.`, "neutral");
  }

  async function queueIndexJob(documentIds?: string[]) {
    try {
      const job = await startIndexJob(documentIds);
      setActiveJob(job);
      setActivePanel("documents");
      setComposerMenuOpen(false);
      await loadWorkspace({ preserveStatus: true });
      setWorkspaceStatus(buildJobMessage(job), "neutral");
    } catch {
      setWorkspaceStatus("Could not start the indexing job.", "error");
    }
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(event.target.files ?? []);
    if (selectedFiles.length === 0) {
      return;
    }

    setUploadBusy(true);
    setComposerMenuOpen(false);
    setWorkspaceStatus(`Uploading ${selectedFiles.length} document${selectedFiles.length === 1 ? "" : "s"}...`, "neutral");

    try {
      const uploaded = await uploadDocuments(selectedFiles);
      await loadWorkspace({ preserveStatus: true });
      await queueIndexJob(uploaded.documents.map((document) => document.id));
      setWorkspaceStatus(
        `${uploaded.documents.length} document${uploaded.documents.length === 1 ? "" : "s"} uploaded. Background indexing started.`,
        "success"
      );
    } catch {
      setWorkspaceStatus("Upload failed. Check the file size and document format.", "error");
    } finally {
      setUploadBusy(false);
      event.target.value = "";
    }
  }

  async function handleIndexPending() {
    if (pendingDocuments.length === 0) {
      setWorkspaceStatus("No pending documents to index.", "neutral");
      return;
    }

    await queueIndexJob(pendingDocuments.map((document) => document.id));
  }

  async function handleSend(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const trimmed = composer.trim();
    if (!trimmed || chatBusy) {
      return;
    }

    setChatBusy(true);
    setWorkspaceStatus("Retrieving context and generating a DeepSeek response...", "neutral");

    let sessionId = activeSessionId;
    if (!sessionId) {
      try {
        const created = await createSession();
        sessionId = created.session_id;
        setActiveSessionId(created.session_id);
      } catch {
        setWorkspaceStatus("Could not create a session for this conversation.", "error");
        setChatBusy(false);
        return;
      }
    }

    const optimisticUserMessage: MessageRecord = {
      role: "user",
      content: trimmed,
      timestamp: new Date().toISOString()
    };

    setMessages((current) => [...current, optimisticUserMessage]);
    setComposer("");

    try {
      const response = await sendChatQuery({
        query: trimmed,
        session_id: sessionId ?? undefined,
        use_fusion: false
      });

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: response.response,
          timestamp: new Date().toISOString()
        }
      ]);
      setSources(response.sources);
      await loadWorkspace({ preserveStatus: true });
      setWorkspaceStatus("Answer generated from indexed sources.", "success");
    } catch {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: "Generation failed. Check DeepSeek validation and retry the query.",
          timestamp: new Date().toISOString()
        }
      ]);
      setWorkspaceStatus("Chat query failed.", "error");
    } finally {
      setChatBusy(false);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  }

  return (
    <div className={`workspace-shell ${sidebarOpen ? "sidebar-open" : ""} ${desktopRailOpen ? "" : "rail-hidden"}`}>
        <aside className="left-rail" aria-label="Conversation history">
          <div className="left-rail__header">
            <div className="brand-lockup">
              <div className="brand-mark">R</div>
              <div>
                <p className="eyebrow">Enterprise RAG</p>
                <h1>RAGagument</h1>
              </div>
            </div>

            <button
              aria-label="Close sidebar"
              className="icon-button mobile-only"
              onClick={() => setSidebarOpen(false)}
              type="button"
            >
              <X size={18} />
            </button>

            <button
              aria-label="Collapse sidebar"
              className="icon-button desktop-only"
              onClick={() => setDesktopRailOpen(false)}
              type="button"
            >
              <ChevronLeft size={18} />
            </button>
          </div>

          <button className="primary-button" onClick={handleNewChat} type="button">
            <Plus size={16} />
            New chat
          </button>

          <label className="search-field">
            <Search size={16} />
            <input
              aria-label="Search chat history"
              onChange={(event) => setSessionSearch(event.target.value)}
              placeholder="Search chat history"
              type="search"
              value={sessionSearch}
            />
          </label>

          <button
            aria-expanded={historyExpanded}
            className="section-toggle"
            onClick={() => setHistoryExpanded((current) => !current)}
            type="button"
          >
            <span className="section-toggle__label">
              <span>Chats</span>
              <span>{sessions.length}</span>
            </span>
            {historyExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </button>

          {historyExpanded ? (
            <div className="session-list" role="list">
              {filteredSessions.length === 0 ? (
                <div className="empty-rail-card">
                  {sessions.length === 0 ? "No chat history yet." : "No sessions match this search. Try a shorter term."}
                </div>
              ) : (
                filteredSessions.map((session) => (
                  <button
                    className={`session-card ${session.session_id === activeSessionId ? "active" : ""}`}
                    key={session.session_id}
                    onClick={() => void openSession(session.session_id)}
                    type="button"
                  >
                    <span className="session-card__title">{session.title}</span>
                    <span className="session-card__time">{formatDateTime(session.start_time)}</span>
                  </button>
                ))
              )}
            </div>
          ) : null}

          <div className="left-rail__footer">
            <div className="rail-shortcuts">
              <button
                className={`rail-shortcut ${activePanel === "documents" ? "active" : ""}`}
                onClick={() => togglePanel("documents")}
                type="button"
              >
                <FolderOpen size={16} />
                <span>Library</span>
              </button>

              <button
                className={`rail-shortcut ${activePanel === "sources" ? "active" : ""}`}
                disabled={sources.length === 0}
                onClick={() => togglePanel("sources")}
                type="button"
              >
                <FileStack size={16} />
                <span>Sources</span>
              </button>
            </div>

            <button
              className={`account-button ${activePanel === "settings" ? "active" : ""}`}
              onClick={() => togglePanel("settings")}
              type="button"
            >
              <span className="account-avatar">{userInitials}</span>
              <span className="account-copy">
                <strong>{profileName}</strong>
                <span>{userEmail}</span>
              </span>
              <ShieldCheck size={16} />
            </button>
          </div>
        </aside>

        <button
          aria-hidden={!sidebarOpen}
          className="rail-backdrop"
          onClick={() => setSidebarOpen(false)}
          tabIndex={sidebarOpen ? 0 : -1}
          type="button"
        />

        <section className="chat-stage">
          {!desktopRailOpen ? (
            <button
              aria-label="Expand sidebar"
              className="icon-button rail-fab desktop-only"
              onClick={() => setDesktopRailOpen(true)}
              type="button"
            >
              <ChevronRight size={18} />
            </button>
          ) : null}

          <button
            aria-label="Open sidebar"
            className="icon-button rail-fab mobile-only"
            onClick={() => setSidebarOpen(true)}
            type="button"
          >
            <Menu size={18} />
          </button>

          {activeJob ? (
            <section className={`job-banner ${activeJob.status}`}>
              <div className="job-banner__icon">
                {activeJob.status === "completed" ? (
                  <CheckCircle2 size={18} />
                ) : activeJob.status === "failed" ? (
                  <X size={18} />
                ) : (
                  <LoaderCircle className="spin" size={18} />
                )}
              </div>

              <div className="job-banner__copy">
                <strong>{buildJobMessage(activeJob)}</strong>
                <span>
                  Submitted {formatDateTime(activeJob.submitted_at)}
                  {activeJob.finished_at ? ` · finished ${formatDateTime(activeJob.finished_at)}` : ""}
                </span>
              </div>

              <button className="text-button" onClick={() => setActivePanel("documents")} type="button">
                Review documents
              </button>
            </section>
          ) : null}

          {activePanel ? (
            <>
              <button
                aria-label="Close side panel"
                className="panel-backdrop"
                onClick={closePanel}
                type="button"
              />

              <aside className="context-drawer inspector-panel">
              {activePanel === "documents" ? (
                <>
                  <div className="context-drawer__header">
                    <div>
                      <h3>Document library</h3>
                      <p>
                        {indexedDocuments.length} indexed · {pendingDocuments.length} pending
                      </p>
                    </div>

                    <div className="context-drawer__controls">
                      <button
                        className="toolbar-button"
                        disabled={pendingDocuments.length === 0 || jobIsRunning}
                        onClick={() => void handleIndexPending()}
                        type="button"
                      >
                        {jobIsRunning ? <LoaderCircle className="spin" size={16} /> : <Database size={16} />}
                        Index pending
                      </button>

                      <button className="icon-button" onClick={closePanel} type="button">
                        <X size={16} />
                      </button>
                    </div>
                  </div>

                  <div className="document-grid">
                    {documents.length === 0 ? (
                      <div className="drawer-card muted-card">
                        Upload PDF, DOCX, TXT, or Markdown files to begin building the corpus.
                      </div>
                    ) : (
                      documents.map((document) => (
                        <article className="drawer-card" key={document.id}>
                          <div className="drawer-card__row">
                            <strong>{document.original_filename}</strong>
                            <span className={`status-badge ${document.processing_status}`}>
                              {documentStatusLabel(document)}
                            </span>
                          </div>
                          <p>{formatFileSize(document.size_bytes)} · {document.indexed_chunk_count} chunks</p>
                          <p className="drawer-card__meta">
                            {document.last_error
                              ? document.last_error
                              : document.last_indexed_at
                                ? `Last indexed ${formatDateTime(document.last_indexed_at)}`
                                : "Waiting for indexing"}
                          </p>
                        </article>
                      ))
                    )}
                  </div>
                </>
              ) : activePanel === "sources" ? (
                <>
                  <div className="context-drawer__header">
                    <div>
                      <h3>Source traces</h3>
                      <p>Evidence returned for the latest answer.</p>
                    </div>

                    <button className="icon-button" onClick={closePanel} type="button">
                      <X size={16} />
                    </button>
                  </div>

                  <div className="document-grid">
                    {sources.length === 0 ? (
                      <div className="drawer-card muted-card">Ask a question to inspect supporting passages.</div>
                    ) : (
                      sources.map((source, index) => (
                        <article className="drawer-card" key={`${source.filename}-${index}`}>
                          <div className="drawer-card__row">
                            <strong>{source.filename}</strong>
                            <span className="status-badge neutral">{source.score.toFixed(2)}</span>
                          </div>
                          <p>{source.snippet}</p>
                        </article>
                      ))
                    )}
                  </div>
                </>
              ) : (
                <>
                  <div className="context-drawer__header">
                    <div>
                      <h3>User settings</h3>
                      <p>Account, privacy, and EU data handling preferences belong here.</p>
                    </div>

                    <button className="icon-button" onClick={closePanel} type="button">
                      <X size={16} />
                    </button>
                  </div>

                  <div className="settings-grid">
                    <article className="settings-card">
                      <h4>Account</h4>
                      <label className="settings-field">
                        <span>Display name</span>
                        <input
                          onChange={(event) => setProfileName(event.target.value)}
                          type="text"
                          value={profileName}
                        />
                      </label>

                      <label className="settings-field">
                        <span>Work email</span>
                        <div className="settings-input">
                          <Mail size={15} />
                          <input
                            onChange={(event) => setUserEmail(event.target.value)}
                            type="email"
                            value={userEmail}
                          />
                        </div>
                      </label>
                    </article>

                    <article className="settings-card">
                      <h4>Security</h4>
                      <p className="settings-copy">
                        Keep security actions in one place, the way users expect from ChatGPT-style account settings.
                      </p>

                      <div className="settings-actions">
                        <button
                          className="settings-action"
                          onClick={() => handleSettingsAction("Password change")}
                          type="button"
                        >
                          <LockKeyhole size={16} />
                          Change password
                        </button>

                        <button
                          className="settings-action"
                          onClick={() => handleSettingsAction("Session review")}
                          type="button"
                        >
                          <UserRound size={16} />
                          Manage sessions
                        </button>

                        <button
                          className="settings-action"
                          onClick={() => handleSettingsAction("Privacy export")}
                          type="button"
                        >
                          <ShieldCheck size={16} />
                          Export personal data
                        </button>
                      </div>
                    </article>

                    <article className="settings-card">
                      <h4>Privacy & compliance</h4>
                      <div className="settings-field-grid">
                        <label className="settings-field">
                          <span>Data region</span>
                          <select onChange={(event) => setJurisdiction(event.target.value)} value={jurisdiction}>
                            <option>EU / EEA</option>
                            <option>Norway / EFTA</option>
                            <option>Customer-defined</option>
                          </select>
                        </label>

                        <label className="settings-field">
                          <span>Retention</span>
                          <select
                            onChange={(event) => setRetentionPolicy(event.target.value)}
                            value={retentionPolicy}
                          >
                            <option>30 days</option>
                            <option>90 days</option>
                            <option>Customer-defined</option>
                          </select>
                        </label>
                      </div>

                      <label className="settings-field">
                        <span>Policy mode</span>
                        <select onChange={(event) => setComplianceMode(event.target.value)} value={complianceMode}>
                          <option>GDPR strict</option>
                          <option>EU standard</option>
                          <option>DPA review</option>
                        </select>
                      </label>

                      <div className="settings-metrics">
                        <div className="settings-metric">
                          <span>Data region</span>
                          <strong>{jurisdiction}</strong>
                        </div>

                        <div className="settings-metric">
                          <span>Upload cap</span>
                          <strong>{settings?.upload_max_file_size_mb ?? 250} MB</strong>
                        </div>
                      </div>
                    </article>
                  </div>
                </>
              )}
              </aside>
            </>
          ) : null}

          <main className={`chat-main ${isEmptySession ? "empty-session" : ""}`} id="chat-main">
            <div className={`message-stream ${isEmptySession ? "empty" : ""}`}>
              {isEmptySession ? (
                <section className="empty-state">
                  <h3>Hi, {profileName}.</h3>
                  <p>
                    Use the plus button to upload files, open the library, or inspect source traces. When your documents are indexed, ask your first question here.
                  </p>
                </section>
              ) : (
                messages.map((message, index) => {
                  const isAssistant = message.role === "assistant";
                  const isLatestAssistant = isAssistant && index === messages.length - 1 && sources.length > 0;

                  return (
                    <div className={`message-row ${message.role}`} key={`${message.timestamp}-${index}`}>
                      <div className={`message-avatar ${message.role}`}>
                        {isAssistant ? <Bot size={16} /> : <UserRound size={16} />}
                      </div>

                      <article className={`message-bubble ${message.role}`}>
                        <div className="message-meta">
                          <strong>{isAssistant ? "RAGagument" : "You"}</strong>
                          <span>{formatDateTime(message.timestamp)}</span>
                        </div>
                        <p>{message.content}</p>
                        {isLatestAssistant ? (
                          <button
                            className="text-button"
                            onClick={() => setActivePanel("sources")}
                            type="button"
                          >
                            Review source traces
                          </button>
                        ) : null}
                      </article>
                    </div>
                  );
                })
              )}
            </div>

            <form className={`composer-shell ${isEmptySession ? "new-session" : ""}`} onSubmit={(event) => void handleSend(event)}>
              <input
                className="sr-only"
                id="workspace-upload"
                multiple
                onChange={handleUpload}
                type="file"
              />

              <div className={`composer-surface ${isEmptySession ? "new-session" : ""}`} ref={composerSurfaceRef}>
                {composerMenuOpen ? (
                  <div className={`composer-menu ${isEmptySession ? "composer-menu--drop" : ""}`} role="menu">
                    <label
                      className="composer-menu__item"
                      htmlFor="workspace-upload"
                      onClick={() => setComposerMenuOpen(false)}
                    >
                      {uploadBusy ? <LoaderCircle className="spin" size={16} /> : <Paperclip size={16} />}
                      <span>
                        <strong>Upload files</strong>
                        <small>PDF, DOCX, TXT, Markdown</small>
                      </span>
                    </label>

                    <button
                      className="composer-menu__item"
                      onClick={() => togglePanel("documents")}
                      type="button"
                    >
                      <FolderOpen size={16} />
                      <span>
                        <strong>Open library</strong>
                        <small>{health?.document_count ?? 0} documents</small>
                      </span>
                    </button>

                    <button
                      className="composer-menu__item"
                      disabled={sources.length === 0}
                      onClick={() => togglePanel("sources")}
                      type="button"
                    >
                      <FileStack size={16} />
                      <span>
                        <strong>View sources</strong>
                        <small>{sources.length} traces from the last answer</small>
                      </span>
                    </button>
                  </div>
                ) : null}

                <div className="composer-row">
                  <button
                    aria-expanded={composerMenuOpen}
                    aria-label="Open upload and library options"
                    className={`composer-plus ${composerMenuOpen ? "active" : ""}`}
                    onClick={() => setComposerMenuOpen((current) => !current)}
                    type="button"
                  >
                    <Plus size={18} />
                  </button>

                  <label className="composer-field">
                    <textarea
                      onChange={(event) => setComposer(event.target.value)}
                      onFocus={() => setComposerMenuOpen(false)}
                      onKeyDown={handleComposerKeyDown}
                      placeholder="Ask across indexed policy packs, contracts, security standards, or operating documents..."
                      rows={1}
                      value={composer}
                    />
                  </label>

                  <button className="send-button" disabled={chatBusy || !composer.trim()} type="submit">
                    {chatBusy ? <LoaderCircle className="spin" size={16} /> : <Send size={16} />}
                  </button>
                </div>

                <div className="composer-statusbar" aria-label="Corpus status">
                  <span className="composer-pill">
                    <FolderOpen size={13} />
                    {health?.document_count ?? 0} docs
                  </span>
                  <span className="composer-pill">
                    <Database size={13} />
                    {health?.indexed_chunk_count ?? 0} chunks
                  </span>
                  <span className={`composer-pill ${jobIsRunning ? "busy" : pendingDocuments.length > 0 ? "pending" : "ready"}`}>
                    {jobIsRunning ? <LoaderCircle className="spin" size={13} /> : <CheckCircle2 size={13} />}
                    {composerStatus}
                  </span>
                </div>
              </div>
            </form>
          </main>
        </section>
      </div>
  );
}
