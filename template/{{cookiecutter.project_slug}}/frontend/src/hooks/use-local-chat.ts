"use client";

import { useCallback, useState } from "react";
import { nanoid } from "nanoid";
import { useWebSocket } from "./use-websocket";
import { useLocalChatStore } from "@/stores/local-chat-store";
import type { ChatMessage, ToolCall, WSEvent } from "@/types";
import { WS_URL } from "@/lib/constants";

export function useLocalChat() {
  const {
    currentConversationId,
    getCurrentMessages,
    createConversation,
    addMessage,
    updateMessage,
    addToolCall,
    updateToolCall,
    clearCurrentMessages,
  } = useLocalChatStore();

  const messages = getCurrentMessages();
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentMessageId, setCurrentMessageId] = useState<string | null>(null);

  const handleWebSocketMessage = useCallback(
    (event: MessageEvent) => {
      const wsEvent: WSEvent = JSON.parse(event.data);

      switch (wsEvent.type) {
        case "model_request_start": {
          const newMsgId = nanoid();
          setCurrentMessageId(newMsgId);
          addMessage({
            id: newMsgId,
            role: "assistant",
            content: "",
            timestamp: new Date(),
            isStreaming: true,
            toolCalls: [],
          });
          break;
        }

        case "text_delta": {
          if (currentMessageId) {
            const content = (wsEvent.data as { index: number; content: string })
              .content;
            updateMessage(currentMessageId, (msg) => ({
              ...msg,
              content: msg.content + content,
            }));
          }
          break;
        }

        case "tool_call": {
          if (currentMessageId) {
            const { tool_name, args, tool_call_id } = wsEvent.data as {
              tool_name: string;
              args: Record<string, unknown>;
              tool_call_id: string;
            };
            const toolCall: ToolCall = {
              id: tool_call_id,
              name: tool_name,
              args,
              status: "running",
            };
            addToolCall(currentMessageId, toolCall);
          }
          break;
        }

        case "tool_result": {
          if (currentMessageId) {
            const { tool_call_id, content } = wsEvent.data as {
              tool_call_id: string;
              content: string;
            };
            updateToolCall(currentMessageId, tool_call_id, {
              result: content,
              status: "completed",
            });
          }
          break;
        }

        case "final_result": {
          if (currentMessageId) {
            updateMessage(currentMessageId, (msg) => ({
              ...msg,
              isStreaming: false,
            }));
          }
          setIsProcessing(false);
          setCurrentMessageId(null);
          break;
        }

        case "error": {
          if (currentMessageId) {
            updateMessage(currentMessageId, (msg) => ({
              ...msg,
              content: msg.content + "\n\n[Error occurred]",
              isStreaming: false,
            }));
          }
          setIsProcessing(false);
          break;
        }

        case "complete": {
          setIsProcessing(false);
          break;
        }
      }
    },
    [currentMessageId, addMessage, updateMessage, addToolCall, updateToolCall]
  );

  const wsUrl = `${WS_URL}/api/v1/ws/agent`;

  const { isConnected, connect, disconnect, sendMessage } = useWebSocket({
    url: wsUrl,
    onMessage: handleWebSocketMessage,
  });

  const sendChatMessage = useCallback(
    (content: string) => {
      let convId = currentConversationId;
      if (!convId) {
        convId = createConversation();
      }

      const userMessage: ChatMessage = {
        id: nanoid(),
        role: "user",
        content,
        timestamp: new Date(),
      };
      addMessage(userMessage);

      setIsProcessing(true);
      sendMessage({ message: content });
    },
    [addMessage, sendMessage, currentConversationId, createConversation]
  );

  const startNewChat = useCallback(() => {
    createConversation();
  }, [createConversation]);

  return {
    messages,
    currentConversationId,
    isConnected,
    isProcessing,
    connect,
    disconnect,
    sendMessage: sendChatMessage,
    clearMessages: clearCurrentMessages,
    startNewChat,
  };
}
