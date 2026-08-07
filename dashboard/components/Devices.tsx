"use client";

import { useState } from "react";
import {
  ArrowUp,
  RotateCw,
} from "lucide-react";

export default function Devices() {
  const [prompt, setPrompt] = useState(
    ""
  );

  return (
    <div className="scale-100">
      <div className="flex w-40 flex-col gap-3">
        <div className="flex flex-col overflow-hidden rounded-3xl">
        {/* Content */}
            <div className="flex flex-1 flex-col items-center justify-center gap-10 px-6">
            {/* Phone */}
            <div className="flex flex-col items-center gap-2">
                <svg
                className="h-14 w-14 text-white"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.8}
                viewBox="0 0 24 24"
                >
                <rect x="7" y="2" width="10" height="20" rx="2" />
                <line x1="11" y1="18" x2="13" y2="18" />
                </svg>
                <span className="text-sm text-white">Phone</span>
            </div>

            {/* Laptop */}
            <div className="flex flex-col items-center gap-2">
                <svg
                className="h-16 w-16 text-white"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.8}
                viewBox="0 0 24 24"
                >
                <rect x="4" y="5" width="16" height="11" rx="1.5" />
                <path d="M2 19h20" />
                </svg>
                <span className="text-sm text-white">Laptop</span>
            </div>

            {/* Cloud */}
            <div className="flex flex-col items-center gap-2">
                <svg
                className="h-16 w-16 text-white"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.8}
                viewBox="0 0 24 24"
                >
                <path d="M7 18a4 4 0 0 1-.5-8A5.5 5.5 0 0 1 17 8.5 3.5 3.5 0 1 1 18 18H7z" />
                </svg>
                <span className="text-sm text-white">Cloud</span>
            </div>
            </div>
        </div>
      </div>
    </div>
  );
}