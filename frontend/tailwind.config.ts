import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#171717",
        paper: "#f6f1e8",
        ember: "#b05a36",
        pine: "#1f4d43",
        mist: "#d9e4df"
      },
      fontFamily: {
        display: ["Fraunces", "Georgia", "serif"],
        body: ["IBM Plex Sans", "Segoe UI", "sans-serif"]
      },
      boxShadow: {
        panel: "0 18px 50px rgba(23, 23, 23, 0.10)"
      }
    }
  },
  plugins: []
};

export default config;
