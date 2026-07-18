import { defineConfig } from "vite";

export default defineConfig({
  root: "src",
  base: "./",
  publicDir: false,
  worker: {
    format: "es",
  },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 2500,
  },
});
