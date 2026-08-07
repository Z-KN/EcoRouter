"use client";

import { useState } from "react";
import {
  ArrowUp,
  RotateCw,
} from "lucide-react";

export default function Prompt() {
  const [prompt, setPrompt] = useState(
    ""
  );

  return (
    <div className="scale-80">
      <div className="flex w-97.5 flex-col gap-3">
        <div className="flex flex-col overflow-hidden rounded-3xl border border-white/10 bg-[#171717] shadow-xl">
          {/* Header */}
          <div className="grid grid-cols-[1fr_auto] gap-y-1 border-b border-white/10 px-5 py-4">
            <h1 className="text-base font-semibold text-white">
                EcoRouter Prompt
            </h1>

            <button
              aria-label="Reset conversation"
              className="row-span-2 flex h-8 w-8 items-center justify-center rounded-2xl border border-white/10 text-[#bdbdbd] transition hover:bg-[#2b2b2b]"
            >
              <RotateCw size={16} />
            </button>

            <p className="text-sm text-[#8a8a8a]">
              Prompt EcoRouter to get the demo started!
            </p>
          </div>


          {/* Footer */}
          <div className="flex flex-col gap-2 px-5 pb-5 pt-5">
            <form>
              <div className="rounded-2xl bg-[#262626]">
                {/* Input */}
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={3}
                  className="w-full resize-none bg-transparent px-4 pt-3 text-[14px] leading-6 text-white outline-none placeholder:text-[#8a8a8a]"
                  placeholder="Ask anything..."
                />

                {/* Toolbar */}
                <div className="flex items-center px-3 pb-3">
                  <button
                    type="submit"
                    className="ml-auto flex h-8 w-8 items-center justify-center rounded-xl bg-white text-black transition hover:bg-[#ececec]"
                  >
                    <ArrowUp size={16} strokeWidth={2.5} />
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>

        <p className="px-2 text-center text-xs text-[#7a7a7a]">
          Demo is read only. Press send to send messages.
        </p>
      </div>
    </div>
  );
}