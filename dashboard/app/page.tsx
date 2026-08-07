import EcoRouter from "@/components/EcoRouter";
import Devices from "@/components/Devices";
import Prompt from "@/components/Prompt";
import Response from "@/components/Response";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col bg-black px-8 py-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-white">
          EcoRouter Dashboard Demo
        </h1>

        <p className="mt-4 text-lg text-gray-400">
          Interactive routing and optimization preview
        </p>
      </div>

      {/* Dashboard */}
      <div className="flex flex-1 items-center">
        <div className="grid w-full grid-cols-[1fr_auto_240px_1fr] items-center gap-8">
          <div className="flex justify-start">
            <Prompt />
          </div>

          <div className="flex justify-center">
            <EcoRouter />
          </div>

          <div className="flex justify-center">
            <Devices />
          </div>

          <div className="flex justify-end">
            <Response />
          </div>
        </div>
      </div>
    </main>
  );
}