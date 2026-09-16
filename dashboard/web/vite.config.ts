import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// 개발: vite 가 /api 를 FastAPI(8082) 로 프록시. 운영: FastAPI 가 dist/ 를 직접 서빙.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { '/api': 'http://localhost:8082' } },
})
