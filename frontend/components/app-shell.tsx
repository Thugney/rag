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
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Database,
  FileStack,
  FolderPlus,
  FolderOpen,
  LoaderCircle,
  Mail,
  Menu,
  Paperclip,
  PencilLine,
  Plus,
  Search,
  Send,
  ShieldCheck,
  Trash2,
  UserRound,
  X
} from "lucide-react";

import {
  ChatResponse,
  deleteDocument,
  deleteProject,
  DocumentRecord,
  HealthResponse,
  IndexJobResponse,
  LLMValidationResponse,
  MetadataValue,
  MessageRecord,
  ProjectSummary,
  SessionSummary,
  SettingsResponse,
  createProject,
  createSession,
  getDocuments,
  getHealth,
  getIndexJob,
  getProjects,
  getSessionMessages,
  getSessions,
  getSettings,
  sendChatQuery,
  startIndexJob,
  updateProject,
  uploadDocuments,
  validateLLM
} from "../lib/api";

type PanelKey = "documents" | "sources" | "settings" | null;
type StatusTone = "neutral" | "success" | "error";

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

function formatMetadataValue(value: MetadataValue): string {
  if (Array.isArray(value)) {
    return value.map((item) => formatMetadataValue(item)).join(", ");
  }

  if (value && typeof value === "object") {
    return Object.entries(value)
      .map(([key, entryValue]) => `${key}: ${formatMetadataValue(entryValue)}`)
      .join(", ");
  }

  return String(value);
}

function formatDocumentParserSummary(document: DocumentRecord): string {
  const parts: string[] = [];

  if ((document.parser_names?.length ?? 0) > 0) {
    parts.push(document.parser_names!.join(", "));
  }

  if ((document.parsed_element_count ?? 0) > 0) {
    parts.push(`${document.parsed_element_count} elements`);
  }

  if ((document.detected_content_types?.length ?? 0) > 0) {
    parts.push(document.detected_content_types!.join(" / "));
  }

  return parts.join(" · ");
}

function formatParserLabel(value: string): string {
  const labels: Record<string, string> = {
    pdf: "PDF",
    docx: "DOCX",
    markdown: "Markdown",
    text: "Text",
    image: "OCR image",
    spreadsheet: "Spreadsheet"
  };
  return labels[value] ?? value;
}

function formatContentTypeLabel(value: string): string {
  const labels: Record<string, string> = {
    text: "Text",
    table: "Table",
    image: "Image OCR",
    figure: "Figure",
    image_reference: "Image ref"
  };
  return labels[value] ?? value;
}

function inferDocumentParseMode(document: DocumentRecord): string {
  const parsers = new Set(document.parser_names ?? []);
  const contentTypes = new Set(document.detected_content_types ?? []);

  if (parsers.has("spreadsheet")) {
    return "Spreadsheet";
  }
  if (parsers.has("image")) {
    return "OCR image";
  }
  if (parsers.has("pdf") && contentTypes.has("image")) {
    return "Scanned PDF OCR";
  }
  if (parsers.has("pdf")) {
    return "Structured PDF";
  }
  if (parsers.has("docx")) {
    return "DOCX";
  }
  if (parsers.has("markdown")) {
    return "Markdown";
  }
  if (parsers.has("text")) {
    return "Text";
  }
  return "Document";
}

function buildDocumentBadges(document: DocumentRecord): string[] {
  const badges: string[] = [inferDocumentParseMode(document)];

  for (const contentType of document.detected_content_types ?? []) {
    badges.push(formatContentTypeLabel(contentType));
  }

  if ((document.parsed_element_count ?? 0) > 0) {
    badges.push(`${document.parsed_element_count} elements`);
  }

  return badges.slice(0, 4);
}

function formatSourceContext(source: ChatResponse["sources"][number]): string {
  const labels: Record<string, string> = {
    page_number: "Page",
    sheet_name: "Sheet",
    structural_role: "Role",
    content_type: "Type",
    parser_name: "Parser",
    element_index: "Element",
    heading_level: "Heading",
    layout_mode: "Layout",
    ocr_source: "OCR",
    chunk_strategy: "Chunk"
  };

  const orderedKeys = [
    "page_number",
    "sheet_name",
    "row_start",
    "row_end",
    "structural_role",
    "content_type",
    "parser_name",
    "chunk_strategy",
    "layout_mode",
    "ocr_source",
    "element_index",
    "heading_level"
  ];

  const parts = orderedKeys.flatMap((key) => {
    const value = source.metadata[key];
    if (value === undefined || value === null || value === "") {
      return [];
    }

    return [`${labels[key] ?? key}: ${formatMetadataValue(value)}`];
  });

  return parts.slice(0, 4).join(" · ");
}

function buildSourceBadges(source: ChatResponse["sources"][number]): string[] {
  const badges: string[] = [];
  const contentType = source.metadata.content_type;
  const parserName = source.metadata.parser_name;
  const chunkStrategy = source.metadata.chunk_strategy;
  const pageNumber = source.metadata.page_number;
  const sheetName = source.metadata.sheet_name;
  const rowStart = source.metadata.row_start;
  const rowEnd = source.metadata.row_end;
  const ocrSource = source.metadata.ocr_source;

  if (typeof contentType === "string" && contentType) {
    badges.push(formatContentTypeLabel(contentType));
  }
  if (typeof parserName === "string" && parserName) {
    badges.push(formatParserLabel(parserName));
  }
  if (typeof chunkStrategy === "string" && chunkStrategy) {
    badges.push(chunkStrategy.replace(/_/g, " "));
  }
  if (typeof pageNumber === "number" || typeof pageNumber === "string") {
    badges.push(`Page ${pageNumber}`);
  }
  if (typeof sheetName === "string" && sheetName) {
    badges.push(`Sheet ${sheetName}`);
  }
  if ((typeof rowStart === "number" || typeof rowStart === "string") && (typeof rowEnd === "number" || typeof rowEnd === "string")) {
    badges.push(`Rows ${rowStart}-${rowEnd}`);
  }
  if (typeof ocrSource === "string" && ocrSource) {
    badges.push(ocrSource.replace(/_/g, " "));
  }

  return badges.slice(0, 5);
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
  const [isHydrated, setIsHydrated] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [llmValidation, setLlmValidation] = useState<LLMValidationResponse | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [messages, setMessages] = useState<MessageRecord[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [composer, setComposer] = useState("");
  const [sources, setSources] = useState<ChatResponse["sources"]>([]);
  const [status, setStatus] = useState("Connecting to the workspace.");
  const [statusTone, setStatusTone] = useState<StatusTone>("neutral");
  const [chatBusy, setChatBusy] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [projectBusy, setProjectBusy] = useState(false);
  const [activeJob, setActiveJob] = useState<IndexJobResponse | null>(null);
  const [activePanel, setActivePanel] = useState<PanelKey>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [desktopRailOpen, setDesktopRailOpen] = useState(true);
  const [composerMenuOpen, setComposerMenuOpen] = useState(false);
  const [projectsExpanded, setProjectsExpanded] = useState(true);
  const [historyExpanded, setHistoryExpanded] = useState(true);
  const [projectFormOpen, setProjectFormOpen] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [projectDescription, setProjectDescription] = useState("");
  const [projectDraftName, setProjectDraftName] = useState("");
  const [projectDraftDescription, setProjectDraftDescription] = useState("");
  const [projectSettingsBusy, setProjectSettingsBusy] = useState(false);
  const [projectDeleteBusy, setProjectDeleteBusy] = useState(false);
  const [documentActionId, setDocumentActionId] = useState<string | null>(null);
  const [sessionSearch, setSessionSearch] = useState("");
  const [profileName, setProfileName] = useState("Workspace Admin");
  const [userEmail, setUserEmail] = useState("admin@ragagument.local");

  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const composerSurfaceRef = useRef<HTMLDivElement | null>(null);
  const deferredSessionSearch = useDeferredValue(sessionSearch);

  const activeProject = projects.find((project) => project.project_id === activeProjectId) ?? null;
  const filteredSessions = sessions.filter((session) => {
    const query = deferredSessionSearch.trim().toLowerCase();
    if (!query) {
      return true;
    }

    return session.title.toLowerCase().includes(query) || session.start_time.toLowerCase().includes(query);
  });

  const indexedDocuments = documents.filter((document) => document.processing_status === "indexed");
  const pendingDocuments = documents.filter((document) => document.processing_status !== "indexed");
  const totalChunks = documents.reduce((sum, document) => sum + document.indexed_chunk_count, 0);
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
    setIsHydrated(true);
  }, []);

  useEffect(() => {
    void loadWorkspace();
  }, []);

  useEffect(() => {
    setProjectDraftName(activeProject?.name ?? "");
    setProjectDraftDescription(activeProject?.description ?? "");
  }, [activeProject?.project_id, activeProject?.name, activeProject?.description]);

  useEffect(() => {
    if (!activeProjectId) {
      startTransition(() => {
        setDocuments([]);
        setSessions([]);
        setMessages([]);
        setSources([]);
        setActiveSessionId(null);
        setActiveJob(null);
      });
      return;
    }

    void loadProjectScope(activeProjectId, true);
  }, [activeProjectId]);

  useEffect(() => {
    if (!activeJob || !["queued", "running"].includes(activeJob.status)) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void pollIndexJob(activeJob.job_id);
    }, 2000);

    return () => window.clearInterval(intervalId);
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
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [composerMenuOpen]);

  function setWorkspaceStatus(message: string, tone: StatusTone = "neutral") {
    setStatus(message);
    setStatusTone(tone);
  }

  async function loadWorkspace(preferredProjectId?: string | null) {
    try {
      const [healthData, settingsData, projectsData] = await Promise.all([
        getHealth(),
        getSettings(),
        getProjects()
      ]);
      const llmData = await validateLLM().catch(() => null);

      const nextProjectId =
        preferredProjectId && projectsData.some((project) => project.project_id === preferredProjectId)
          ? preferredProjectId
          : activeProjectId && projectsData.some((project) => project.project_id === activeProjectId)
            ? activeProjectId
            : projectsData[0]?.project_id ?? null;

      startTransition(() => {
        setHealth(healthData);
        setSettings(settingsData);
        setLlmValidation(llmData);
        setProjects(projectsData);
        if (nextProjectId !== activeProjectId) {
          setActiveProjectId(nextProjectId);
          setActiveSessionId(null);
          setMessages([]);
          setSources([]);
        }
      });

      if (!nextProjectId) {
        setWorkspaceStatus("Create a project to group its documents and chats together.", "neutral");
      } else if (nextProjectId === activeProjectId) {
        await loadProjectScope(nextProjectId, true);
      } else {
        setWorkspaceStatus("Project workspace ready.", llmData?.valid === false ? "error" : "success");
      }
    } catch {
      setWorkspaceStatus("API not reachable. Start the FastAPI service to continue.", "error");
    }
  }

  async function loadProjectScope(projectId: string, preserveStatus = false) {
    try {
      const [documentsData, sessionsData] = await Promise.all([
        getDocuments(projectId),
        getSessions(projectId)
      ]);

      const runningJobId = documentsData.find((document) => document.active_index_job_id)?.active_index_job_id;
      const currentJob = runningJobId ? await getIndexJob(runningJobId) : null;

      startTransition(() => {
        setDocuments(documentsData);
        setSessions(sessionsData);
        setActiveJob(currentJob);
      });

      if (!sessionsData.some((session) => session.session_id === activeSessionId)) {
        startTransition(() => {
          setActiveSessionId(null);
          setMessages([]);
          setSources([]);
        });
      }

      if (!preserveStatus) {
        setWorkspaceStatus("Project loaded.", "neutral");
      }
    } catch {
      setWorkspaceStatus("Could not load the selected project.", "error");
    }
  }

  async function pollIndexJob(jobId: string) {
    try {
      const job = await getIndexJob(jobId);
      setActiveJob(job);

      if (job.status === "completed" || job.status === "failed") {
        await loadWorkspace(job.project_id);
        setWorkspaceStatus(buildJobMessage(job), job.status === "completed" ? "success" : "error");
      }
    } catch {
      setWorkspaceStatus("Could not refresh indexing status.", "error");
    }
  }

  async function openSession(session: SessionSummary) {
    try {
      if (session.project_id !== activeProjectId) {
        setActiveProjectId(session.project_id);
      }

      const history = await getSessionMessages(session.session_id);
      startTransition(() => {
        setActiveSessionId(session.session_id);
        setMessages(history);
        setSources([]);
      });
      setSidebarOpen(false);
      setWorkspaceStatus("Conversation history loaded.", "neutral");
    } catch {
      setWorkspaceStatus("Could not load session history.", "error");
    }
  }

  async function handleCreateProject(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedName = projectName.trim();
    if (!trimmedName || projectBusy) {
      return;
    }

    setProjectBusy(true);
    try {
      const created = await createProject({
        name: trimmedName,
        description: projectDescription.trim()
      });
      setProjectFormOpen(false);
      setProjectName("");
      setProjectDescription("");
      await loadWorkspace(created.project_id);
      setWorkspaceStatus(`Project "${created.name}" created.`, "success");
    } catch {
      setWorkspaceStatus("Could not create the project.", "error");
    } finally {
      setProjectBusy(false);
    }
  }

  function openCreateProjectFlow() {
    setComposerMenuOpen(false);
    setProjectFormOpen(true);
    setDesktopRailOpen(true);
    setSidebarOpen(true);
    setProjectName("");
    setProjectDescription("");
    setWorkspaceStatus("Create a project, then upload the documents that belong to it.", "neutral");
  }

  function togglePanel(panel: Exclude<PanelKey, null>) {
    if (panel === "documents" && !activeProject) {
      setWorkspaceStatus("Create or select a project first.", "error");
      return;
    }
    setComposerMenuOpen(false);
    setActivePanel((current) => (current === panel ? null : panel));
  }

  function openProjectSettings() {
    if (!activeProject) {
      setWorkspaceStatus("Create or select a project first.", "error");
      return;
    }

    setProjectDraftName(activeProject.name);
    setProjectDraftDescription(activeProject.description);
    setComposerMenuOpen(false);
    setActivePanel("settings");
  }

  async function handleUpdateProject() {
    if (!activeProject) {
      setWorkspaceStatus("Create or select a project first.", "error");
      return;
    }

    const trimmedName = projectDraftName.trim();
    if (!trimmedName || projectSettingsBusy) {
      return;
    }

    setProjectSettingsBusy(true);
    try {
      const updated = await updateProject(activeProject.project_id, {
        name: trimmedName,
        description: projectDraftDescription.trim()
      });
      await loadWorkspace(updated.project_id);
      setWorkspaceStatus(`Project "${updated.name}" updated.`, "success");
    } catch {
      setWorkspaceStatus("Could not update the project.", "error");
    } finally {
      setProjectSettingsBusy(false);
    }
  }

  async function handleDeleteProject() {
    if (!activeProject) {
      setWorkspaceStatus("Create or select a project first.", "error");
      return;
    }

    const confirmed = window.confirm(
      `Delete project "${activeProject.name}"? This removes its chats and uploaded documents.`
    );
    if (!confirmed) {
      return;
    }

    setProjectDeleteBusy(true);
    try {
      const projectNameToDelete = activeProject.name;
      await deleteProject(activeProject.project_id);
      setActivePanel(null);
      setActiveSessionId(null);
      await loadWorkspace(null);
      setWorkspaceStatus(`Project "${projectNameToDelete}" deleted.`, "success");
    } catch {
      setWorkspaceStatus("Could not delete the project.", "error");
    } finally {
      setProjectDeleteBusy(false);
    }
  }

  async function handleDeleteDocument(document: DocumentRecord) {
    const confirmed = window.confirm(`Remove "${document.original_filename}" from this project?`);
    if (!confirmed) {
      return;
    }

    setDocumentActionId(document.id);
    try {
      await deleteDocument(document.project_id, document.id);
      await loadWorkspace(document.project_id);
      setWorkspaceStatus(`Removed "${document.original_filename}" from the project.`, "success");
    } catch {
      setWorkspaceStatus("Could not remove the document from the project.", "error");
    } finally {
      setDocumentActionId(null);
    }
  }

  function handleNewChat() {
    if (!activeProjectId) {
      setWorkspaceStatus("Create or select a project first.", "error");
      return;
    }

    startTransition(() => {
      setActiveSessionId(null);
      setMessages([]);
      setSources([]);
    });
    setWorkspaceStatus("New chat ready inside the current project.", "neutral");
  }

  async function queueIndexJob(documentIds?: string[]) {
    if (!activeProjectId) {
      setWorkspaceStatus("Create or select a project first.", "error");
      return;
    }

    try {
      const job = await startIndexJob(activeProjectId, documentIds);
      setActiveJob(job);
      setActivePanel("documents");
      setComposerMenuOpen(false);
      await loadWorkspace(activeProjectId);
      setWorkspaceStatus(buildJobMessage(job), "neutral");
    } catch {
      setWorkspaceStatus("Could not start the indexing job.", "error");
    }
  }

  function openFilePicker() {
    setComposerMenuOpen(false);
    window.setTimeout(() => uploadInputRef.current?.click(), 0);
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(event.target.files ?? []);
    if (selectedFiles.length === 0) {
      return;
    }

    if (!activeProjectId) {
      setWorkspaceStatus("Create or select a project before uploading documents.", "error");
      event.target.value = "";
      return;
    }

    setUploadBusy(true);
    try {
      const uploaded = await uploadDocuments(activeProjectId, selectedFiles);
      await loadWorkspace(activeProjectId);
      await queueIndexJob(uploaded.documents.map((document) => document.id));
      setWorkspaceStatus("Documents uploaded and queued for indexing.", "success");
    } catch {
      setWorkspaceStatus("Upload failed. Check the file size and document format.", "error");
    } finally {
      setUploadBusy(false);
      event.target.value = "";
    }
  }

  async function handleSend(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const trimmed = composer.trim();
    if (!trimmed || chatBusy) {
      return;
    }

    if (!activeProjectId) {
      setWorkspaceStatus("Create or select a project before starting a chat.", "error");
      return;
    }

    setChatBusy(true);
    let sessionId = activeSessionId;
    if (!sessionId) {
      try {
        const created = await createSession(activeProjectId);
        sessionId = created.session_id;
        setActiveSessionId(created.session_id);
      } catch {
        setWorkspaceStatus("Could not create a session for this chat.", "error");
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
        project_id: activeProjectId,
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
      await loadWorkspace(activeProjectId);
      setWorkspaceStatus("Answer generated from project-scoped sources.", "success");
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

  if (!isHydrated) {
    return null;
  }

  return (
    <div className={`workspace-shell ${sidebarOpen ? "sidebar-open" : ""} ${desktopRailOpen ? "" : "rail-hidden"}`}>
      <aside className="left-rail" aria-label="Project navigation">
        <div className="left-rail__header">
          <div className="brand-lockup">
            <div className="brand-mark">R</div>
            <div>
              <p className="eyebrow">Enterprise RAG</p>
              <h1>RAGagument</h1>
            </div>
          </div>

          <button className="icon-button mobile-only" onClick={() => setSidebarOpen(false)} type="button">
            <X size={18} />
          </button>

          <button className="icon-button desktop-only" onClick={() => setDesktopRailOpen(false)} type="button">
            <ChevronLeft size={18} />
          </button>
        </div>

        <button className="primary-button" onClick={handleNewChat} type="button">
          <Plus size={16} />
          New chat
        </button>

        <button className="secondary-button" onClick={() => setProjectFormOpen((current) => !current)} type="button">
          <FolderOpen size={16} />
          {projectFormOpen ? "Close project form" : "New project"}
        </button>

        <label className="search-field">
          <Search size={16} />
          <input
            aria-label="Search chat history"
            onChange={(event) => setSessionSearch(event.target.value)}
            placeholder="Search chats in this project"
            type="search"
            value={sessionSearch}
          />
        </label>

        <div className="left-rail__content">
          {projectFormOpen ? (
            <form className="project-form" onSubmit={(event) => void handleCreateProject(event)}>
              <label className="field-input">
                <FolderOpen size={16} />
                <input
                  onChange={(event) => setProjectName(event.target.value)}
                  placeholder="Project name"
                  type="text"
                  value={projectName}
                />
              </label>
              <label className="project-form__description">
                <textarea
                  onChange={(event) => setProjectDescription(event.target.value)}
                  placeholder="What belongs in this project?"
                  rows={3}
                  value={projectDescription}
                />
              </label>
              <div className="project-form__actions">
                <button className="toolbar-button" disabled={projectBusy || !projectName.trim()} type="submit">
                  {projectBusy ? <LoaderCircle className="spin" size={16} /> : <Plus size={16} />}
                  Create project
                </button>
              </div>
            </form>
          ) : null}

          <button className="section-toggle" onClick={() => setProjectsExpanded((current) => !current)} type="button">
            <span className="section-toggle__label">
              <span>Projects</span>
              <span>{projects.length}</span>
            </span>
            {projectsExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </button>

          {projectsExpanded ? (
            <div className="project-list" role="list">
              {projects.length === 0 ? (
                <div className="empty-rail-card">Create your first project to group documents and chats together.</div>
              ) : (
                projects.map((project) => (
                  <button
                    className={`project-card ${project.project_id === activeProjectId ? "active" : ""}`}
                    key={project.project_id}
                    onClick={() => setActiveProjectId(project.project_id)}
                    type="button"
                  >
                    <span className="session-card__title">{project.name}</span>
                    {project.description ? <span className="project-card__description">{project.description}</span> : null}
                    <span className="project-card__meta">
                      {project.document_count} docs · {project.session_count} chats
                    </span>
                  </button>
                ))
              )}
            </div>
          ) : null}

          <button className="section-toggle" onClick={() => setHistoryExpanded((current) => !current)} type="button">
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
                  {sessions.length === 0 ? "No chats in this project yet." : "No chats match this search."}
                </div>
              ) : (
                filteredSessions.map((session) => (
                  <button
                    className={`session-card ${session.session_id === activeSessionId ? "active" : ""}`}
                    key={session.session_id}
                    onClick={() => void openSession(session)}
                    type="button"
                  >
                    <span className="session-card__title">{session.title}</span>
                    <span className="session-card__time">{formatDateTime(session.start_time)}</span>
                  </button>
                ))
              )}
            </div>
          ) : null}
        </div>

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
              onClick={() => togglePanel("sources")}
              type="button"
            >
              <FileStack size={16} />
              <span>Sources</span>
            </button>
          </div>

          <button
            className={`account-button ${activePanel === "settings" ? "active" : ""}`}
            onClick={() => setActivePanel(activePanel === "settings" ? null : "settings")}
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

      <button className="rail-backdrop" onClick={() => setSidebarOpen(false)} tabIndex={sidebarOpen ? 0 : -1} type="button" />

      <section className="chat-stage">
        {!desktopRailOpen ? (
          <button className="icon-button rail-fab desktop-only" onClick={() => setDesktopRailOpen(true)} type="button">
            <ChevronRight size={18} />
          </button>
        ) : null}

        <button className="icon-button rail-fab mobile-only" onClick={() => setSidebarOpen(true)} type="button">
          <Menu size={18} />
        </button>
        {activeJob ? (
          <section className={`job-banner ${activeJob.status}`}>
            <div className="job-banner__icon">
              {activeJob.status === "completed" ? <CheckCircle2 size={18} /> : <LoaderCircle className="spin" size={18} />}
            </div>
            <div className="job-banner__copy">
              <strong>{buildJobMessage(activeJob)}</strong>
              <span>{activeProject?.name ?? "Project"} · {formatDateTime(activeJob.submitted_at)}</span>
            </div>
          </section>
        ) : null}

        {activePanel ? (
          <>
            <button className="panel-backdrop" onClick={() => setActivePanel(null)} type="button" />
            <aside className="context-drawer inspector-panel">
              {activePanel === "documents" ? (
                <>
                  <div className="context-drawer__header">
                    <div>
                      <h3>{activeProject ? `${activeProject.name} library` : "Document library"}</h3>
                      <p>{indexedDocuments.length} indexed · {pendingDocuments.length} pending</p>
                    </div>
                    <div className="context-drawer__controls">
                      <button
                        className="toolbar-button"
                        disabled={!activeProjectId || pendingDocuments.length === 0 || jobIsRunning}
                        onClick={() => void queueIndexJob(pendingDocuments.map((document) => document.id))}
                        type="button"
                      >
                        {jobIsRunning ? <LoaderCircle className="spin" size={16} /> : <Database size={16} />}
                        Index pending
                      </button>
                      <button className="icon-button" onClick={() => setActivePanel(null)} type="button">
                        <X size={16} />
                      </button>
                    </div>
                  </div>

                  <div className="document-grid">
                    {documents.length === 0 ? (
                      <div className="drawer-card muted-card">Upload documents into this project to build its corpus.</div>
                    ) : (
                      documents.map((document) => (
                        <article className="drawer-card" key={document.id}>
                          <div className="drawer-card__row">
                            <strong>{document.original_filename}</strong>
                            <div className="drawer-card__actions">
                              <span className={`status-badge ${document.processing_status}`}>{documentStatusLabel(document)}</span>
                              <button
                                aria-label={`Remove ${document.original_filename}`}
                                className="icon-button icon-button--danger icon-button--small"
                                disabled={documentActionId === document.id}
                                onClick={() => void handleDeleteDocument(document)}
                                type="button"
                              >
                                {documentActionId === document.id ? <LoaderCircle className="spin" size={14} /> : <Trash2 size={14} />}
                              </button>
                            </div>
                          </div>
                          <p>{formatFileSize(document.size_bytes)} · {document.indexed_chunk_count} chunks</p>
                          <div className="drawer-card__badges">
                            {buildDocumentBadges(document).map((badge) => (
                              <span className="meta-badge" key={`${document.id}-${badge}`}>
                                {badge}
                              </span>
                            ))}
                          </div>
                          {formatDocumentParserSummary(document) ? (
                            <p className="drawer-card__meta">{formatDocumentParserSummary(document)}</p>
                          ) : null}
                          {document.last_error ? (
                            <div className="drawer-card__alert">{document.last_error}</div>
                          ) : (
                            <p className="drawer-card__meta">
                              {document.last_indexed_at
                                ? `Last indexed ${formatDateTime(document.last_indexed_at)}`
                                : "Waiting for indexing"}
                            </p>
                          )}
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
                      <p>Evidence returned for the latest answer in this project.</p>
                    </div>
                    <button className="icon-button" onClick={() => setActivePanel(null)} type="button">
                      <X size={16} />
                    </button>
                  </div>

                  <div className="document-grid">
                    {sources.length === 0 ? (
                      <div className="drawer-card muted-card">Ask a question to inspect the supporting passages.</div>
                    ) : (
                      sources.map((source, index) => (
                        <article className="drawer-card" key={`${source.filename}-${index}`}>
                          <div className="drawer-card__row">
                            <strong>{source.filename}</strong>
                            <span className="status-badge neutral">{source.score.toFixed(2)}</span>
                          </div>
                          <p>{source.snippet}</p>
                          <div className="drawer-card__badges">
                            {buildSourceBadges(source).map((badge) => (
                              <span className="meta-badge" key={`${source.filename}-${index}-${badge}`}>
                                {badge}
                              </span>
                            ))}
                          </div>
                          {formatSourceContext(source) ? (
                            <p className="drawer-card__meta">{formatSourceContext(source)}</p>
                          ) : null}
                        </article>
                      ))
                    )}
                  </div>
                </>
              ) : (
                <>
                  <div className="context-drawer__header">
                    <div>
                      <h3>Workspace settings</h3>
                      <p>Manage the active project here, then keep auth and user profile below it.</p>
                    </div>
                    <button className="icon-button" onClick={() => setActivePanel(null)} type="button">
                      <X size={16} />
                    </button>
                  </div>
                  <div className="settings-grid">
                    <article className="settings-card">
                      <h4>Current project</h4>
                      {activeProject ? (
                        <>
                          <label className="settings-field">
                            <span>Project name</span>
                            <input
                              onChange={(event) => setProjectDraftName(event.target.value)}
                              type="text"
                              value={projectDraftName}
                            />
                          </label>
                          <label className="settings-field">
                            <span>Description</span>
                            <textarea
                              onChange={(event) => setProjectDraftDescription(event.target.value)}
                              rows={4}
                              value={projectDraftDescription}
                            />
                          </label>
                          <div className="settings-actions">
                            <button
                              className="settings-action"
                              disabled={projectSettingsBusy || !projectDraftName.trim()}
                              onClick={() => void handleUpdateProject()}
                              type="button"
                            >
                              {projectSettingsBusy ? <LoaderCircle className="spin" size={16} /> : <PencilLine size={16} />}
                              Save project
                            </button>
                            <button
                              className="settings-action danger"
                              disabled={projectDeleteBusy}
                              onClick={() => void handleDeleteProject()}
                              type="button"
                            >
                              {projectDeleteBusy ? <LoaderCircle className="spin" size={16} /> : <Trash2 size={16} />}
                              Delete project
                            </button>
                          </div>
                        </>
                      ) : (
                        <p className="settings-copy">Create or select a project first.</p>
                      )}
                    </article>
                    <article className="settings-card">
                      <h4>Account</h4>
                      <label className="settings-field">
                        <span>Display name</span>
                        <input onChange={(event) => setProfileName(event.target.value)} type="text" value={profileName} />
                      </label>
                      <label className="settings-field">
                        <span>Work email</span>
                        <div className="settings-input">
                          <Mail size={15} />
                          <input onChange={(event) => setUserEmail(event.target.value)} type="email" value={userEmail} />
                        </div>
                      </label>
                    </article>
                    <article className="settings-card">
                      <h4>Auth plan</h4>
                      <p className="settings-copy">
                        Recommended path: Supabase Auth with Postgres first. Start with email and password, reset flows,
                        magic links, and project ownership. Add enterprise SSO after the core project workflow is stable.
                      </p>
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
                <h3>{activeProject ? `Hi, ${activeProject.name}.` : `Hi, ${profileName}.`}</h3>
                <p>
                  {activeProject
                    ? activeProject.description || "Upload the documents that belong to this project, then keep every related chat inside the same project."
                    : "Create a project, upload the relevant documents, and keep each related chat inside that project."}
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
                        <button className="text-button" onClick={() => setActivePanel("sources")} type="button">
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
            <input ref={uploadInputRef} className="sr-only" multiple onChange={handleUpload} type="file" />

            <div className={`composer-surface ${isEmptySession ? "new-session" : ""}`} ref={composerSurfaceRef}>
              {composerMenuOpen ? (
                <div className={`composer-menu ${isEmptySession ? "composer-menu--drop" : ""}`} role="menu">
                  <button className="composer-menu__item" onClick={openCreateProjectFlow} type="button">
                    <FolderPlus size={16} />
                    <span>
                      <strong>New project</strong>
                      <small>Create a project workspace with its own documents and chats</small>
                    </span>
                  </button>
                  <button className="composer-menu__item" onClick={openFilePicker} type="button">
                    {uploadBusy ? <LoaderCircle className="spin" size={16} /> : <Paperclip size={16} />}
                    <span>
                      <strong>Upload files</strong>
                      <small>{activeProject ? `Into ${activeProject.name}` : "Select a project first"}</small>
                    </span>
                  </button>
                  <button className="composer-menu__item" disabled={!activeProject} onClick={openProjectSettings} type="button">
                    <PencilLine size={16} />
                    <span>
                      <strong>Project settings</strong>
                      <small>{activeProject ? `Edit ${activeProject.name} and manage its documents` : "Select a project first"}</small>
                    </span>
                  </button>
                  <button className="composer-menu__item" onClick={() => togglePanel("documents")} type="button">
                    <FolderOpen size={16} />
                    <span>
                      <strong>Open library</strong>
                      <small>{documents.length} documents in this project</small>
                    </span>
                  </button>
                  <button className="composer-menu__item" onClick={() => togglePanel("sources")} type="button">
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
                    placeholder={activeProject ? `Ask about ${activeProject.name}...` : "Create a project before chatting..."}
                    rows={1}
                    value={composer}
                  />
                </label>
                <button className="send-button" disabled={chatBusy || !composer.trim() || !activeProjectId} type="submit">
                  {chatBusy ? <LoaderCircle className="spin" size={16} /> : <Send size={16} />}
                </button>
              </div>

              <div className="composer-statusbar" aria-label="Project status">
                <span className="composer-pill">
                  <FolderOpen size={13} />
                  {activeProject?.name ?? "No project"}
                </span>
                <span className="composer-pill">
                  <Database size={13} />
                  {documents.length} docs · {totalChunks} chunks
                </span>
                <span className={`composer-pill ${jobIsRunning ? "busy" : pendingDocuments.length > 0 ? "pending" : "ready"}`}>
                  {jobIsRunning ? <LoaderCircle className="spin" size={13} /> : <CheckCircle2 size={13} />}
                  {composerStatus}
                </span>
              </div>
            </div>
          </form>

          <div className={`status-banner ${statusTone}`}>
            <span>{status}</span>
            {health ? <span>{health.project_count} projects · {health.document_count} docs overall</span> : null}
            {llmValidation && !llmValidation.valid ? <span>{llmValidation.message}</span> : null}
          </div>
        </main>
      </section>
    </div>
  );
}
