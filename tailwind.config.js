/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: '#0B0F19',
        surface: '#111827',
        'surface-elevated': '#1F2937',
        'surface-border': '#374151',
        primary: {
          DEFAULT: '#6366F1', // Indigo
          hover: '#4F46E5',
          light: '#818CF8',
        },
        secondary: {
          DEFAULT: '#8B5CF6', // Purple
          hover: '#7C3AED',
          light: '#A78BFA',
        },
        accent: '#06B6D4', // Cyan
      },
    },
  },
  plugins: [],
}
