"use client";

export default function EcoRouter() {
  return (
    <div className="scale-100">
      <div className="flex w-80 flex-col gap-3">
        <div className="flex flex-col overflow-hidden rounded-3xl border border-white/10 bg-[#171717] shadow-xl">
          <div className="grid grid-cols-[1fr_auto] gap-y-1  px-5 py-4">
            <h1 className="text-base font-semibold text-white">
              EcoRouter Program
            </h1>
            <p className="text-sm text-[#8a8a8a]">
              Waiting on prompt...
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}