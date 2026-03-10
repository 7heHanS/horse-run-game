import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
    base: '/horse-run-game/',
    plugins: [
        viteStaticCopy({
            targets: [
                {
                    src: 'node_modules/onnxruntime-web/dist/*.wasm',
                    dest: ''
                }
            ]
        })
    ],
    server: {
        port: 3000,
        open: true,
        fs: {
            strict: false 
        }
    },
    optimizeDeps: {
        exclude: ['onnxruntime-web'] 
    }
});
