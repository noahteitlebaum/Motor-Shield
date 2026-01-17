import React from "react";

interface MotorHealthBarProps {
  health: number; // 0 to 100
}

const MotorHealthBar: React.FC<MotorHealthBarProps> = ({ health }) => {
  // Clamp value between 0 and 100
  const clampedHealth = Math.min(100, Math.max(0, health));

  // Dynamic color calculation for smoothness
  // Map 0-100 to Hue 0 (Red) - 120 (Green)
  // Saturation 90%, Lightness 45% for nice visibility
  const hue = (clampedHealth / 100) * 120;
  const colorStyle = `hsl(${hue}, 90%, 45%)`;
  const shadowStyle = `0 0 15px hsl(${hue}, 90%, 45%, 0.6)`;

  return (
    <div className="w-full max-w-md mx-auto p-4 bg-white/50 dark:bg-[var(--card-bg)]/50 backdrop-blur-md rounded-2xl border border-[var(--card-border)] shadow-xl transition-colors duration-300">
      <div className="flex justify-between items-center mb-2">
        <span className="text-[var(--text-secondary)] font-medium tracking-wide">
          Motor Health
        </span>
        <span className="text-[var(--text-primary)] font-bold">
          {Math.round(clampedHealth)}%
        </span>
      </div>

      {/* Background Track */}
      <div className="h-4 w-full bg-gray-200 dark:bg-gray-700/50 rounded-full overflow-hidden backdrop-blur-sm border border-black/5 dark:border-gray-600/30">
        {/* Active Bar */}
        <div
          className="h-full transition-all duration-700 ease-in-out rounded-full relative"
          style={{
            width: `${clampedHealth}%`,
            backgroundColor: colorStyle,
            boxShadow: shadowStyle,
          }}
        >
          {/* Shine effect */}
          <div className="absolute top-0 left-0 w-full h-1/2 bg-white/20 rounded-t-full"></div>

          {/* Subtle Texture/Gradient overlay for depth */}
          <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/10"></div>
        </div>
      </div>
    </div>
  );
};

export default MotorHealthBar;
