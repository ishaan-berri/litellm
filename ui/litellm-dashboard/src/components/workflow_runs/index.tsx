import React, { useCallback, useEffect, useRef, useState } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { Button, Collapse, message, Space, Tag, Tooltip, Typography } from "antd";
import {
  CheckOutlined,
  CloseOutlined,
  CopyOutlined,
  DownOutlined,
  ReloadOutlined,
  UpOutlined,
} from "@ant-design/icons";
import { DataTable } from "../view_logs/table";
import { DrawerShell } from "@/components/shared/DrawerShell";
import { useItemNavigation } from "@/components/shared/useItemNavigation";
import { proxyBaseUrl } from "@/components/networking";

const { Text } = Typography;

// ── types ────────────────────────────────────────────────────────────────────

interface WorkflowRunsProps {
  accessToken: string | null;
}

type RunStatus = "pending" | "running" | "paused" | "completed" | "failed";

interface RunMetadata {
  title?: string;
  state?: string;
  pr_url?: string;
  worktree_path?: string;
  plan_text?: string;
  grill_session_id?: string;
  session_id?: string;
  [key: string]: unknown;
}

interface WorkflowRun {
  run_id: string;
  status: RunStatus;
  workflow_type: string;
  created_at: string;
  metadata?: RunMetadata | null;
}

interface WorkflowRunEvent {
  event_id: string;
  event_type: string;
  step_name: string;
  sequence_number: number;
  created_at: string;
  data?: Record<string, unknown> | null;
}

interface WorkflowRunMessage {
  message_id: string;
  role: string;
  content: string;
  sequence_number: number;
  created_at: string;
}

// ── design tokens ─────────────────────────────────────────────────────────────

const STATUS_COLOR: Record<RunStatus, string> = {
  pending:   "#a1a1aa",
  running:   "#3b82f6",
  paused:    "#f59e0b",
  completed: "#22c55e",
  failed:    "#ef4444",
};

const STATUS_TAG_COLOR: Record<RunStatus, string> = {
  pending:   "default",
  running:   "processing",
  paused:    "gold",
  completed: "success",
  failed:    "error",
};

const EVENT_COLOR: Record<string, { bar: string; border: string; text: string }> = {
  "step.started":  { bar: "#f0fdf4", border: "#86efac", text: "#16a34a" },
  "step.failed":   { bar: "#fef2f2", border: "#fca5a5", text: "#dc2626" },
  "hook.waiting":  { bar: "#fffbeb", border: "#fcd34d", text: "#d97706" },
  "hook.received": { bar: "#eff6ff", border: "#93c5fd", text: "#2563eb" },
};

function eventStyle(type: string) {
  return EVENT_COLOR[type] ?? { bar: "#f4f4f5", border: "#d4d4d8", text: "#52525b" };
}

// ── helpers ───────────────────────────────────────────────────────────────────

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  if (isNaN(diff)) return iso;
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function fmtDuration(ms: number): string {
  if (ms < 0) return "";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function runTitle(run: WorkflowRun): string {
  const t = run.metadata?.title;
  if (t) return String(t);
  return run.workflow_type ?? run.run_id.slice(0, 8);
}

// ── status dot ────────────────────────────────────────────────────────────────

const StatusDot: React.FC<{ status: RunStatus; size?: number }> = ({ status, size = 8 }) => (
  <span
    style={{
      display: "inline-block",
      width: size,
      height: size,
      borderRadius: "50%",
      background: STATUS_COLOR[status] ?? "#a1a1aa",
      flexShrink: 0,
    }}
  />
);

// ── truncated value ───────────────────────────────────────────────────────────

const TRUNCATE_AT = 120;

const TruncatedValue: React.FC<{ value: string }> = ({ value }) => {
  const [expanded, setExpanded] = useState(false);
  if (value.length <= TRUNCATE_AT) {
    return <span style={{ color: "#27272a", wordBreak: "break-all" }}>{value}</span>;
  }
  return (
    <span style={{ color: "#27272a", wordBreak: "break-all" }}>
      {expanded ? value : value.slice(0, TRUNCATE_AT) + "…"}
      <button
        onClick={() => setExpanded((e) => !e)}
        style={{
          background: "none",
          border: "none",
          padding: "0 4px",
          cursor: "pointer",
          color: "#2563eb",
          fontSize: 11,
        }}
      >
        {expanded ? "less" : "more"}
      </button>
    </span>
  );
};

// ── field pair ────────────────────────────────────────────────────────────────

const FieldPair: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
    <span style={{ fontSize: 10, color: "#a1a1aa", textTransform: "uppercase", letterSpacing: "0.06em" }}>
      {label}
    </span>
    <span style={{ fontSize: 12 }}>{children}</span>
  </div>
);

// ── metadata card ─────────────────────────────────────────────────────────────

const MetadataCard: React.FC<{ run: WorkflowRun }> = ({ run }) => {
  const meta = run.metadata ?? {};
  const primaryFields: { key: string; label: string }[] = [
    { key: "state",            label: "state" },
    { key: "worktree_path",    label: "worktree" },
    { key: "grill_session_id", label: "grill session" },
    { key: "session_id",       label: "session" },
  ];
  const primaryKeys = new Set(["title", ...primaryFields.map((f) => f.key)]);
  const extraEntries = Object.entries(meta).filter(
    ([k, v]) => !primaryKeys.has(k) && v !== null && v !== undefined && v !== ""
  );

  return (
    <div style={{ borderRadius: 8, border: "1px solid #e4e4e7", marginBottom: 16, overflow: "hidden" }}>
      <div style={{ padding: "14px 20px", borderBottom: "1px solid #f4f4f5", display: "flex", alignItems: "center", gap: 10 }}>
        <StatusDot status={run.status} size={10} />
        <span style={{ fontSize: 14, fontWeight: 600, color: "#18181b", flex: 1 }}>
          {runTitle(run)}
        </span>
        <span style={{ fontFamily: "monospace", fontSize: 11, color: "#a1a1aa", background: "#f4f4f5", padding: "2px 8px", borderRadius: 4 }}>
          {run.run_id.slice(0, 8)}
        </span>
        <span style={{ fontSize: 11, color: "#a1a1aa", background: "#f4f4f5", padding: "2px 8px", borderRadius: 4 }}>
          {run.workflow_type}
        </span>
      </div>
      <div style={{ padding: "12px 20px", display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "8px 24px", fontFamily: "monospace", fontSize: 12 }}>
        <FieldPair label="status">
          <span style={{ textTransform: "capitalize", color: "#27272a" }}>{run.status}</span>
        </FieldPair>
        <FieldPair label="created">
          <span style={{ color: "#27272a" }}>{timeAgo(run.created_at)}</span>
        </FieldPair>
        {meta.pr_url && (
          <FieldPair label="pr">
            <a href={String(meta.pr_url)} target="_blank" rel="noopener noreferrer" style={{ color: "#2563eb", textDecoration: "none", wordBreak: "break-all" }}>
              {String(meta.pr_url)}
            </a>
          </FieldPair>
        )}
        {primaryFields.map(({ key, label }) => {
          const v = meta[key];
          if (v === null || v === undefined || v === "") return null;
          const str = typeof v === "object" ? JSON.stringify(v) : String(v);
          return (
            <FieldPair key={key} label={label}>
              <TruncatedValue value={str} />
            </FieldPair>
          );
        })}
        {extraEntries.map(([k, v]) => {
          const str = typeof v === "object" ? JSON.stringify(v) : String(v);
          return (
            <FieldPair key={k} label={k}>
              <TruncatedValue value={str} />
            </FieldPair>
          );
        })}
      </div>
    </div>
  );
};

// ── gantt timeline ────────────────────────────────────────────────────────────

const GanttTimeline: React.FC<{ run: WorkflowRun; events: WorkflowRunEvent[] }> = ({ run, events }) => {
  if (events.length === 0) {
    return <div style={{ padding: "16px 0", color: "#a1a1aa", fontSize: 12, fontFamily: "monospace" }}>No events recorded</div>;
  }

  const runStart = new Date(run.created_at).getTime();
  const eventTimes = events.map((e) => new Date(e.created_at).getTime());
  const lastTime = Math.max(...eventTimes);
  const totalSpan = Math.max(lastTime - runStart, 1);
  const totalDur = fmtDuration(lastTime - runStart);

  return (
    <div style={{ fontFamily: "monospace", fontSize: 12 }}>
      <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "0 12px", marginBottom: 2 }}>
        <div />
        <div style={{ position: "relative", height: 16 }}>
          {[0, 100].map((pct) => (
            <span key={pct} style={{ position: "absolute", left: `${pct}%`, transform: pct === 100 ? "translateX(-100%)" : undefined, fontSize: 10, color: "#a1a1aa" }}>
              {pct === 0 ? "0" : totalDur}
            </span>
          ))}
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "0 12px", marginBottom: 4 }}>
        <div style={{ color: "#3f3f46", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", paddingTop: 2 }}>
          {runTitle(run)}
        </div>
        <div style={{ height: 24, background: "#f4f4f5", border: "1px solid #d4d4d8", borderRadius: 4, display: "flex", alignItems: "center", paddingLeft: 8 }}>
          <span style={{ color: "#71717a", fontSize: 11 }}>{totalDur}</span>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "0 12px", rowGap: 3 }}>
        {events.map((ev) => {
          const evTime = new Date(ev.created_at).getTime();
          const leftPct = ((evTime - runStart) / totalSpan) * 100;
          const nextIdx = events.findIndex((e) => e.sequence_number > ev.sequence_number);
          const nextTime = nextIdx >= 0 ? new Date(events[nextIdx].created_at).getTime() : lastTime + Math.max(totalSpan * 0.12, 500);
          const widthPct = Math.max(8, ((nextTime - evTime) / totalSpan) * 100);
          const style = eventStyle(ev.event_type);
          const dur = fmtDuration(nextTime - evTime);

          return (
            <React.Fragment key={ev.event_id}>
              <div style={{ color: style.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", paddingTop: 2, paddingLeft: 12 }}>
                {ev.step_name || ev.event_type}
              </div>
              <div style={{ position: "relative", height: 24 }}>
                <Tooltip
                  title={
                    <div style={{ fontFamily: "monospace", fontSize: 11, lineHeight: 1.6 }}>
                      <div><span style={{ color: "#a1a1aa" }}>type: </span><span style={{ color: style.text }}>{ev.event_type}</span></div>
                      <div><span style={{ color: "#a1a1aa" }}>step: </span>{ev.step_name}</div>
                      <div><span style={{ color: "#a1a1aa" }}>seq:  </span>{ev.sequence_number}</div>
                      <div><span style={{ color: "#a1a1aa" }}>time: </span>{timeAgo(ev.created_at)}</div>
                      {ev.data && Object.keys(ev.data).length > 0 && (
                        <div><span style={{ color: "#a1a1aa" }}>data: </span>{JSON.stringify(ev.data)}</div>
                      )}
                    </div>
                  }
                >
                  <div
                    style={{
                      position: "absolute",
                      left: `${Math.min(leftPct, 92)}%`,
                      width: `${Math.min(widthPct, 100 - Math.min(leftPct, 92))}%`,
                      height: "100%",
                      background: style.bar,
                      border: `1px solid ${style.border}`,
                      borderRadius: 4,
                      display: "flex",
                      alignItems: "center",
                      paddingLeft: 8,
                      cursor: "default",
                      overflow: "hidden",
                      gap: 6,
                    }}
                  >
                    <span style={{ color: style.text, whiteSpace: "nowrap", fontSize: 11 }}>{ev.event_type}</span>
                    {dur && <span style={{ color: "#a1a1aa", whiteSpace: "nowrap", fontSize: 11 }}>{dur}</span>}
                  </div>
                </Tooltip>
              </div>
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};

// ── message row ───────────────────────────────────────────────────────────────

const MessageRow: React.FC<{ msg: WorkflowRunMessage }> = ({ msg }) => {
  const roleColor: Record<string, string> = {
    user: "#2563eb", assistant: "#16a34a", system: "#7c3aed", tool_result: "#d97706",
  };
  return (
    <div style={{ display: "grid", gridTemplateColumns: "80px 1fr", gap: "0 16px", padding: "10px 0", borderBottom: "1px solid #f4f4f5", fontFamily: "monospace", fontSize: 12, alignItems: "start" }}>
      <span style={{ color: roleColor[msg.role] ?? "#52525b", paddingTop: 1 }}>[{msg.role}]</span>
      <div>
        <span style={{ color: "#27272a", lineHeight: 1.6, whiteSpace: "pre-wrap", wordBreak: "break-word", display: "block" }}>
          {msg.content}
        </span>
        <span style={{ color: "#a1a1aa", fontSize: 11, marginTop: 2, display: "block" }}>{timeAgo(msg.created_at)}</span>
      </div>
    </div>
  );
};

// ── workflow drawer header ────────────────────────────────────────────────────

const WorkflowDrawerHeader: React.FC<{
  run: WorkflowRun;
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
}> = ({ run, onClose, onPrev, onNext }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(run.run_id);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch { /* clipboard unavailable */ }
  };

  const kbStyle: React.CSSProperties = {
    border: "1px solid #d9d9d9",
    borderRadius: 4,
    padding: "0 4px",
    fontSize: 12,
    fontFamily: "monospace",
    marginLeft: 4,
    background: "#fafafa",
  };

  return (
    <div style={{ padding: "16px 24px", borderBottom: "1px solid #f0f0f0", background: "#fff", position: "sticky", top: 0, zIndex: 10 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
        <Text strong style={{ fontSize: 14, color: "#18181b" }}>{runTitle(run)}</Text>
        <Space size={4} split={<div style={{ width: 1, height: 20, background: "#f0f0f0" }} />}>
          <Button type="text" size="small" onClick={onPrev}>
            <UpOutlined />
            <span style={kbStyle}>K</span>
          </Button>
          <Button type="text" size="small" onClick={onNext}>
            <DownOutlined />
            <span style={kbStyle}>J</span>
          </Button>
          <Tooltip title="ESC to close">
            <Button type="text" icon={<CloseOutlined />} onClick={onClose} />
          </Tooltip>
        </Space>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <Text
          style={{ fontFamily: "monospace", fontSize: 12, color: "#71717a", cursor: "pointer" }}
          onClick={handleCopy}
        >
          {run.run_id}
        </Text>
        <button type="button" onClick={handleCopy} style={{ background: "none", border: "none", cursor: "pointer", color: "#a1a1aa", padding: 0 }}>
          {copied ? <CheckOutlined style={{ fontSize: 11 }} /> : <CopyOutlined style={{ fontSize: 11 }} />}
        </button>
        <Tag color={STATUS_TAG_COLOR[run.status] ?? "default"} style={{ marginLeft: 4, textTransform: "capitalize" }}>
          {run.status}
        </Tag>
        <Tag style={{ fontFamily: "monospace" }}>{run.workflow_type}</Tag>
        <Text type="secondary" style={{ fontSize: 12 }}>{timeAgo(run.created_at)}</Text>
      </div>
    </div>
  );
};

// ── sidebar message row ───────────────────────────────────────────────────────

const MSG_ROLE_COLOR: Record<string, string> = {
  user: "#2563eb",
  assistant: "#16a34a",
  system: "#7c3aed",
  tool_result: "#d97706",
};

const SidebarMessageRow: React.FC<{ msg: WorkflowRunMessage }> = ({ msg }) => (
  <div className="px-3 py-2 border-b border-slate-100 last:border-0">
    <div
      className="text-[10px] font-mono font-semibold mb-0.5"
      style={{ color: MSG_ROLE_COLOR[msg.role] ?? "#52525b" }}
    >
      [{msg.role}]
    </div>
    <div className="text-[11px] text-slate-600 leading-tight line-clamp-2">
      {msg.content.slice(0, 120)}
    </div>
    <div className="text-[10px] text-slate-400 font-mono mt-0.5">{timeAgo(msg.created_at)}</div>
  </div>
);

// ── workflow run drawer ───────────────────────────────────────────────────────

interface WorkflowRunDrawerProps {
  open: boolean;
  onClose: () => void;
  selectedRun: WorkflowRun | null;
  allRuns: WorkflowRun[];
  onSelectRun: (run: WorkflowRun) => void;
  events: WorkflowRunEvent[];
  messages: WorkflowRunMessage[];
  loadingDetail: boolean;
  onRefresh: () => void;
}

const WorkflowRunDrawer: React.FC<WorkflowRunDrawerProps> = ({
  open,
  onClose,
  selectedRun,
  allRuns,
  onSelectRun,
  events,
  messages,
  loadingDetail,
  onRefresh,
}) => {
  const [sidebarCopiedId, setSidebarCopiedId] = useState(false);

  const getRunId = useCallback((run: WorkflowRun) => run.run_id, []);

  const { selectNext, selectPrev } = useItemNavigation({
    isOpen: open,
    currentItem: selectedRun,
    allItems: allRuns,
    getId: getRunId,
    onClose,
    onSelect: onSelectRun,
  });

  const handleCopySidebarId = useCallback(async () => {
    if (!selectedRun) return;
    try {
      await navigator.clipboard.writeText(selectedRun.run_id);
      setSidebarCopiedId(true);
      // reset after 1.2s — using a ref-based cleanup to avoid stale closure
      const t = window.setTimeout(() => setSidebarCopiedId(false), 1200);
      return () => window.clearTimeout(t);
    } catch { /* clipboard unavailable */ }
  }, [selectedRun]);

  const displayId = selectedRun?.run_id ?? "";
  const displayIdShort = displayId.length > 14 ? `${displayId.slice(0, 11)}...` : displayId;

  const sidebarContent = (
    <>
      <div className="pl-12 pr-3 py-2 border-b border-slate-200 bg-white">
        <div className="text-[10px] uppercase tracking-wide text-slate-500">Messages</div>
        <div className="font-mono text-[12px] text-slate-900 leading-tight flex items-center gap-1">
          <span className="truncate">{displayIdShort}</span>
          <button
            type="button"
            onClick={handleCopySidebarId}
            className="text-slate-400 hover:text-slate-600"
            aria-label="Copy run id"
          >
            {sidebarCopiedId ? (
              <CheckOutlined className="text-[11px]" />
            ) : (
              <CopyOutlined className="text-[11px]" />
            )}
          </button>
        </div>
        <div className="mt-1 text-[11px] text-slate-500 font-mono">
          {messages.length} message{messages.length !== 1 ? "s" : ""}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="px-3 py-4 text-[11px] text-slate-400 font-mono">
            {loadingDetail ? "Loading…" : "No messages"}
          </div>
        ) : (
          messages.map((msg) => (
            <SidebarMessageRow key={msg.message_id} msg={msg} />
          ))
        )}
      </div>
    </>
  );

  return (
    <DrawerShell open={open} onClose={onClose} sidebarContent={sidebarContent}>
      {selectedRun && (
        <WorkflowDrawerHeader
          run={selectedRun}
          onClose={onClose}
          onPrev={selectPrev}
          onNext={selectNext}
        />
      )}
      <div className="flex-1 overflow-y-auto">
        {loadingDetail ? (
          <div className="flex items-center justify-center h-32 text-slate-400 text-sm">
            Loading…
          </div>
        ) : selectedRun ? (
          <div style={{ padding: "20px 24px" }}>
            <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
              <Button
                size="small"
                icon={<ReloadOutlined />}
                onClick={onRefresh}
                loading={loadingDetail}
                style={{ color: "#71717a", borderColor: "#e4e4e7" }}
              >
                Refresh
              </Button>
            </div>
            <MetadataCard run={selectedRun} />
            <Collapse
              defaultActiveKey={["timeline"]}
              ghost={false}
              style={{ border: "1px solid #e4e4e7", borderRadius: 8, overflow: "hidden" }}
              items={[
                {
                  key: "timeline",
                  label: (
                    <span style={{ fontSize: 12, fontWeight: 500, color: "#3f3f46" }}>
                      Timeline
                      <span style={{ marginLeft: 6, fontSize: 11, color: "#a1a1aa", fontWeight: 400 }}>
                        {events.length} {events.length === 1 ? "event" : "events"}
                      </span>
                    </span>
                  ),
                  children: (
                    <div style={{ padding: "4px 4px 12px" }}>
                      <GanttTimeline run={selectedRun} events={events} />
                    </div>
                  ),
                },
                {
                  key: "messages",
                  label: (
                    <span style={{ fontSize: 12, fontWeight: 500, color: "#3f3f46" }}>
                      Messages
                      <span style={{ marginLeft: 6, fontSize: 11, color: "#a1a1aa", fontWeight: 400 }}>
                        {messages.length}
                      </span>
                    </span>
                  ),
                  children: messages.length === 0 ? (
                    <div style={{ padding: "12px 4px", color: "#a1a1aa", fontSize: 12, fontFamily: "monospace" }}>
                      No messages
                    </div>
                  ) : (
                    <div style={{ paddingBottom: 4 }}>
                      {messages.map((msg) => (
                        <MessageRow key={msg.message_id} msg={msg} />
                      ))}
                    </div>
                  ),
                },
              ]}
            />
          </div>
        ) : null}
      </div>
    </DrawerShell>
  );
};

// ── TanStack columns for DataTable ────────────────────────────────────────────

const workflowColumns: ColumnDef<WorkflowRun>[] = [
  {
    accessorKey: "run_id",
    header: "Run",
    cell: (info) => {
      const run = info.row.original;
      return (
        <div className="flex items-center gap-2">
          <StatusDot status={run.status} size={7} />
          <div>
            <div className="text-sm font-medium text-slate-900 leading-tight truncate max-w-[28ch]">
              {runTitle(run)}
            </div>
            <div className="font-mono text-[11px] text-slate-400">{run.run_id.slice(0, 8)}</div>
          </div>
        </div>
      );
    },
  },
  {
    accessorKey: "workflow_type",
    header: "Type",
    cell: (info) => (
      <span className="font-mono text-xs text-slate-500">{info.getValue() as string}</span>
    ),
  },
  {
    accessorKey: "status",
    header: "Status",
    cell: (info) => {
      const run = info.row.original;
      const status = info.getValue() as RunStatus;
      const state = run.metadata?.state;
      return (
        <div className="flex items-center gap-1.5">
          <StatusDot status={status} size={7} />
          <span className={`px-2 py-0.5 rounded text-xs font-medium inline-block capitalize ${
            status === "completed" ? "bg-green-50 text-green-700" :
            status === "failed"    ? "bg-red-50 text-red-700" :
            status === "running"   ? "bg-blue-50 text-blue-700" :
            status === "paused"    ? "bg-amber-50 text-amber-700" :
                                     "bg-slate-100 text-slate-600"
          }`}>
            {state ?? status}
          </span>
        </div>
      );
    },
  },
  {
    accessorKey: "created_at",
    header: "Created",
    cell: (info) => (
      <span className="text-xs text-slate-400">{timeAgo(info.getValue() as string)}</span>
    ),
  },
];

// ── main component ────────────────────────────────────────────────────────────

const WorkflowRuns: React.FC<WorkflowRunsProps> = ({ accessToken }) => {
  const [runs, setRuns] = useState<WorkflowRun[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [selectedRun, setSelectedRun] = useState<WorkflowRun | null>(null);
  const [events, setEvents] = useState<WorkflowRunEvent[]>([]);
  const [messages, setMessages] = useState<WorkflowRunMessage[]>([]);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const detailAbortRef = useRef<AbortController | null>(null);

  const fetchRuns = useCallback(async () => {
    if (!accessToken) return;
    setLoadingRuns(true);
    try {
      const res = await fetch(`${proxyBaseUrl ?? ""}/v1/workflows/runs?limit=100`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setRuns(data.runs ?? []);
    } catch (err) {
      console.error("workflow runs fetch failed:", err);
      message.error("Failed to load workflow runs");
    } finally {
      setLoadingRuns(false);
    }
  }, [accessToken]);

  const fetchRunDetail = useCallback(
    async (run: WorkflowRun) => {
      if (!accessToken) return;
      // Cancel any in-flight detail fetch so stale data never overwrites newer selection
      detailAbortRef.current?.abort();
      const controller = new AbortController();
      detailAbortRef.current = controller;

      setSelectedRun(run);
      setDrawerOpen(true);
      setLoadingDetail(true);
      setEvents([]);
      setMessages([]);
      try {
        const base = proxyBaseUrl ?? "";
        const [evRes, msgRes] = await Promise.all([
          fetch(`${base}/v1/workflows/runs/${run.run_id}/events`, {
            headers: { Authorization: `Bearer ${accessToken}` },
            signal: controller.signal,
          }),
          fetch(`${base}/v1/workflows/runs/${run.run_id}/messages`, {
            headers: { Authorization: `Bearer ${accessToken}` },
            signal: controller.signal,
          }),
        ]);
        if (controller.signal.aborted) return;
        const evData = evRes.ok ? await evRes.json() : { events: [] };
        const msgData = msgRes.ok ? await msgRes.json() : { messages: [] };
        setEvents(
          [...(evData.events ?? [])].sort(
            (a: WorkflowRunEvent, b: WorkflowRunEvent) => a.sequence_number - b.sequence_number
          )
        );
        setMessages(
          [...(msgData.messages ?? [])].sort(
            (a: WorkflowRunMessage, b: WorkflowRunMessage) => a.sequence_number - b.sequence_number
          )
        );
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        console.error("workflow run detail fetch failed:", err);
        message.error("Failed to load run details");
      } finally {
        if (!controller.signal.aborted) setLoadingDetail(false);
      }
    },
    [accessToken]
  );

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  const handleSelectRun = useCallback(
    (run: WorkflowRun) => {
      fetchRunDetail(run);
    },
    [fetchRunDetail]
  );

  const handleRefresh = useCallback(() => {
    if (selectedRun) fetchRunDetail(selectedRun);
  }, [selectedRun, fetchRunDetail]);

  return (
    <div className="w-full max-w-screen p-6 overflow-x-hidden box-border" style={{ minHeight: "calc(100vh - 64px)", background: "#fff" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 600, color: "#18181b" }}>Workflow Runs</div>
          <div style={{ fontSize: 13, color: "#71717a", marginTop: 2 }}>
            Durable state tracking for agents and automated workflows
          </div>
        </div>
        <Button
          icon={<ReloadOutlined />}
          onClick={fetchRuns}
          loading={loadingRuns}
          style={{ color: "#71717a", borderColor: "#e4e4e7" }}
        >
          Refresh
        </Button>
      </div>

      <DataTable
        data={runs}
        columns={workflowColumns}
        onRowClick={handleSelectRun}
        isLoading={loadingRuns}
        loadingMessage="Loading workflow runs…"
        noDataMessage="No workflow runs yet"
      />

      <WorkflowRunDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        selectedRun={selectedRun}
        allRuns={runs}
        onSelectRun={handleSelectRun}
        events={events}
        messages={messages}
        loadingDetail={loadingDetail}
        onRefresh={handleRefresh}
      />
    </div>
  );
};

export default WorkflowRuns;
