/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f0fdf4",
          500: "#22c55e",   // dark-mode green (unchanged)
          600: "#16a34a",
          700: "#15803d",
          900: "#14532d",
        },
        // Light-mode "Monochrome + Green" tokens
        mono: {
          bg:         "#f5f5f5",
          surface:    "#ffffff",
          alt:        "#fafafa",
          ink:        "#0a0a0a",
          "ink-muted": "#404040",
          "ink-faint": "#737373",
          "ink-dim":   "#a3a3a3",
          border:     "#e5e5e5",
          divider:    "#f0f0f0",
          green:      "#059669",
          "green-strong": "#047857",
        },
      },
    },
  },
  plugins: [
    function ({ addUtilities }) {
      addUtilities({
        ".no-scrollbar": {
          "-ms-overflow-style": "none",
          "scrollbar-width": "none",
          "&::-webkit-scrollbar": { display: "none" },
        },
      });
    },
  ],
};
