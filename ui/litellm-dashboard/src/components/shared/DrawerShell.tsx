/**
 * DrawerShell — shared layout for detail drawers across the app.
 *
 * Provides the antd Drawer wrapper + a collapsible 224px left sidebar
 * + a flex-1 right main area. Caller supplies sidebar and main content.
 *
 * Used by: LogDetailsDrawer, WorkflowRunDrawer
 */

import React, { useEffect, useState } from "react";
import { Button, Drawer } from "antd";
import { LeftOutlined, RightOutlined } from "@ant-design/icons";

export const DRAWER_SHELL_WIDTH = "60%";
export const DRAWER_SHELL_SIDEBAR_WIDTH = 224;

interface DrawerShellProps {
  open: boolean;
  onClose: () => void;
  width?: string | number;
  /** Content rendered inside the collapsible left sidebar */
  sidebarContent: React.ReactNode;
  /** Content rendered in the right main area (header + scrollable body) */
  children: React.ReactNode;
}

export function DrawerShell({
  open,
  onClose,
  width = DRAWER_SHELL_WIDTH,
  sidebarContent,
  children,
}: DrawerShellProps) {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  useEffect(() => {
    if (open) setIsSidebarCollapsed(false);
  }, [open]);

  return (
    <Drawer
      title={null}
      placement="right"
      onClose={onClose}
      open={open}
      width={width}
      closable={false}
      mask={true}
      maskClosable={true}
      styles={{
        body: { padding: 0, overflow: "hidden" },
        header: { display: "none" },
      }}
    >
      <div style={{ height: "100%" }} className="flex relative">
        <Button
          type="text"
          size="small"
          icon={isSidebarCollapsed ? <RightOutlined /> : <LeftOutlined />}
          onClick={() => setIsSidebarCollapsed((c) => !c)}
          className="absolute top-2 left-2 z-20 !bg-white !border !border-slate-200 !rounded-md"
          aria-label={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        />

        {!isSidebarCollapsed && (
          <div
            className="border-r border-slate-200 bg-slate-50 flex flex-col"
            style={{ width: DRAWER_SHELL_SIDEBAR_WIDTH }}
          >
            {sidebarContent}
          </div>
        )}

        <div className="flex-1 flex flex-col overflow-hidden">
          {children}
        </div>
      </div>
    </Drawer>
  );
}
