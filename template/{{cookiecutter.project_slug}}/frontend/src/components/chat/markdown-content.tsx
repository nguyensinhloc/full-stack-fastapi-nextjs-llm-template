"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { CopyButton } from "./copy-button";

interface MarkdownContentProps {
  content: string;
}

export function MarkdownContent({ content }: MarkdownContentProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={% raw %}{{{% endraw %}
        pre({ children, ...props }) {
          const codeElement = children as React.ReactElement<{
            children?: string;
          }>;
          const codeContent =
            typeof codeElement?.props?.children === "string"
              ? codeElement.props.children
              : "";

          return (
            <div className="group relative">
              <pre
                className="overflow-x-auto rounded-lg bg-muted/50 p-3 text-xs"
                {...props}
              >
                {children}
              </pre>
              {codeContent && (
                <div className="absolute right-2 top-2">
                  <CopyButton text={codeContent} className="opacity-100" />
                </div>
              )}
            </div>
          );
        },
        code({ className, children, ...props }) {
          const isInline = !className;
          if (isInline) {
            return (
              <code
                className="rounded bg-muted px-1.5 py-0.5 text-xs font-mono"
                {...props}
              >
                {children}
              </code>
            );
          }
          return (
            <code className={className} {...props}>
              {children}
            </code>
          );
        },
        a({ href, children, ...props }) {
          return (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline hover:text-primary/80"
              {...props}
            >
              {children}
            </a>
          );
        },
        p({ children, ...props }) {
          return (
            <p className="mb-2 last:mb-0" {...props}>
              {children}
            </p>
          );
        },
        ul({ children, ...props }) {
          return (
            <ul className="mb-2 ml-4 list-disc last:mb-0" {...props}>
              {children}
            </ul>
          );
        },
        ol({ children, ...props }) {
          return (
            <ol className="mb-2 ml-4 list-decimal last:mb-0" {...props}>
              {children}
            </ol>
          );
        },
        li({ children, ...props }) {
          return (
            <li className="mb-1" {...props}>
              {children}
            </li>
          );
        },
        h1({ children, ...props }) {
          return (
            <h1 className="mb-2 text-lg font-bold" {...props}>
              {children}
            </h1>
          );
        },
        h2({ children, ...props }) {
          return (
            <h2 className="mb-2 text-base font-bold" {...props}>
              {children}
            </h2>
          );
        },
        h3({ children, ...props }) {
          return (
            <h3 className="mb-2 text-sm font-bold" {...props}>
              {children}
            </h3>
          );
        },
        blockquote({ children, ...props }) {
          return (
            <blockquote
              className="mb-2 border-l-2 border-muted-foreground/50 pl-3 italic"
              {...props}
            >
              {children}
            </blockquote>
          );
        },
        table({ children, ...props }) {
          return (
            <div className="mb-2 overflow-x-auto">
              <table className="min-w-full text-sm" {...props}>
                {children}
              </table>
            </div>
          );
        },
        th({ children, ...props }) {
          return (
            <th
              className="border-b border-muted px-2 py-1 text-left font-semibold"
              {...props}
            >
              {children}
            </th>
          );
        },
        td({ children, ...props }) {
          return (
            <td className="border-b border-muted/50 px-2 py-1" {...props}>
              {children}
            </td>
          );
        },
        hr({ ...props }) {
          return <hr className="my-3 border-muted" {...props} />;
        },
      {% raw %}}}{% endraw %}
    >
      {content}
    </ReactMarkdown>
  );
}
