/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['Inter', 'sans-serif'],
        body: ['Inter', 'sans-serif'],
      },
      colors: {
        primary: { DEFAULT: '#16a34a', foreground: '#ffffff' },
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
      },
    }
  },
  plugins: [],
}
