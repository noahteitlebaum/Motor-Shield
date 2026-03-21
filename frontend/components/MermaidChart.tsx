"use client";
import React, { useEffect, useState } from "react";
import mermaid from "mermaid";

export default function MermaidChart({ chart }: { chart: string }) {
  const [svg, setSvg] = useState<string>('');

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false,
      theme: "dark",
      securityLevel: "loose",
    });

    const renderChart = async () => {
      try {
        const id = 'mermaid-svg-' + Math.random().toString(36).substring(7);
        const result = await mermaid.render(id, chart);
        setSvg(result.svg);
      } catch (err) {
        console.error("Mermaid parsing failed", err);
      }
    };

    renderChart();
  }, [chart]);

  if (!svg) {
    return (
      <div className="flex justify-center items-center w-full h-[300px] my-6 bg-default-50 rounded-xl border border-default-100 animate-pulse">
        <p className="text-default-400">Rendering Architecture Diagram...</p>
      </div>
    );
  }

  return (
    <div 
      className="flex justify-center w-full overflow-x-auto my-6 p-6 bg-default-50 rounded-xl border border-default-100" 
      dangerouslySetInnerHTML={{ __html: svg }} 
    />
  );
}
