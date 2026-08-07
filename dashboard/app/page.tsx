import Prompt from "@/components/Prompt";
import Response from "@/components/Response";

export default function Home() {
  return (
    <main className="min-h-screen bg-black px-12 py-16">
      <div className="text-center">
        <h1 className="text-6xl font-bold tracking-tight text-white">
          EcoRouter Dashboard Demo
        </h1>

        <p className="mt-4 text-lg text-gray-400">
          Interactive routing and optimization preview
        </p>
      </div>

      <div className="mt-20 grid grid-cols-4 justify-items-center gap-8">
        <Prompt />
        <Prompt />
        <Prompt />
        <Response />
      </div>
    </main>
  );
}