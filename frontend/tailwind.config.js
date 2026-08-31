/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // ChatGPT-like palette
        sidebar: {
          DEFAULT: '#171717',
          hover: '#2a2a2a',
        },
        chat: {
          bg: '#212121',
          input: '#2f2f2f',
          user: '#2f2f2f',
          assistant: '#212121',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
