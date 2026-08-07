"use client";

import { RotateCw, CheckCircle2 } from "lucide-react";

export default function Response() {
  return (
    <div className="flex flex-col gap-3 scale-100">
      <div className="flex flex-col overflow-hidden rounded-3xl border border-white/10 bg-[#171717] shadow-xl">
        {/* Header */}
        <div className="grid grid-cols-[1fr_auto] gap-y-1 border-b border-white/10 px-5 py-4">
          <h1 className="text-base font-semibold text-white">
            EcoRouter Response
          </h1>

          <button
            aria-label="Reset response"
            className="row-span-2 flex h-8 w-8 items-center justify-center rounded-2xl border border-white/10 text-[#bdbdbd] transition hover:bg-[#2b2b2b]"
          >
            <RotateCw size={16} />
          </button>

          <p className="text-sm text-[#8a8a8a]">
            EcoRouter output will appear here.
          </p>
        </div>

        {/* Response Content */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          <div className="rounded-2xl bg-[#262626] p-4">
            <p className="text-sm leading-6 text-[#d6d6d6]">
              Waiting for a prompt...
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-white/10 px-5 py-4">
          <div className="flex items-center gap-2 text-xs text-[#8a8a8a]">
            <CheckCircle2 size={14} className="text-green-500" />
            Ready to receive a response
          </div>
        </div>
      </div>

      <p className="px-2 text-center text-xs text-[#7a7a7a]">
        Responses will stream here after you submit a prompt.
      </p>
    </div>
  );
}