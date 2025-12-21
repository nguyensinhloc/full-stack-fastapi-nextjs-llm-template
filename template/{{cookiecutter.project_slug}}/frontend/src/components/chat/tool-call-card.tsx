"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui";
import type { ToolCall } from "@/types";
import { Wrench, CheckCircle, Loader2, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { CopyButton } from "./copy-button";

interface ToolCallCardProps {
  toolCall: ToolCall;
}

export function ToolCallCard({ toolCall }: ToolCallCardProps) {
  const statusConfig = {
    pending: { icon: Loader2, color: "text-muted-foreground", animate: true },
    running: { icon: Loader2, color: "text-blue-500", animate: true },
    completed: { icon: CheckCircle, color: "text-green-500", animate: false },
    error: { icon: AlertCircle, color: "text-red-500", animate: false },
  };

  const { icon: StatusIcon, color, animate } = statusConfig[toolCall.status];

  const argsText = JSON.stringify(toolCall.args, null, 2);
  const resultText =
    toolCall.result !== undefined
      ? typeof toolCall.result === "string"
        ? toolCall.result
        : JSON.stringify(toolCall.result, null, 2)
      : "";

  return (
    <Card className="bg-muted/50">
      <CardHeader className="py-2 px-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Wrench className="h-4 w-4 text-muted-foreground" />
            <CardTitle className="text-sm font-medium">
              {toolCall.name}
            </CardTitle>
          </div>
          <StatusIcon
            className={cn("h-4 w-4", color, animate && "animate-spin")}
          />
        </div>
      </CardHeader>
      <CardContent className="py-2 px-3 space-y-2">
        {/* Arguments */}
        <div className="group relative">
          <div className="flex items-center justify-between mb-1">
            <p className="text-xs text-muted-foreground">Arguments:</p>
            <CopyButton
              text={argsText}
              className="opacity-0 group-hover:opacity-100"
            />
          </div>
          <pre className="text-xs bg-background p-2 rounded overflow-x-auto">
            {argsText}
          </pre>
        </div>

        {/* Result */}
        {toolCall.result !== undefined && (
          <div className="group relative">
            <div className="flex items-center justify-between mb-1">
              <p className="text-xs text-muted-foreground">Result:</p>
              <CopyButton
                text={resultText}
                className="opacity-0 group-hover:opacity-100"
              />
            </div>
            <pre className="text-xs bg-background p-2 rounded overflow-x-auto max-h-48 overflow-y-auto">
              {resultText}
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
