import { useEffect, useRef } from 'react'

type Props = { src: string; alt: string; overlays: boolean }
export default function ImageViewer({ src, alt, overlays }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas || !overlays) return
    const ctx = canvas.getContext('2d'); if (!ctx) return
    const draw = () => {
      const rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1
      canvas.width = rect.width * ratio; canvas.height = rect.height * ratio; ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
      // Illustrative review regions only; these are not semantic segmentation results.
      const w = rect.width, h = rect.height
      ctx.fillStyle = 'rgba(70,173,222,.19)'; ctx.beginPath(); ctx.moveTo(w*.22,h*.63); ctx.quadraticCurveTo(w*.47,h*.51,w*.72,h*.61); ctx.quadraticCurveTo(w*.54,h*.83,w*.27,h*.76); ctx.closePath(); ctx.fill()
      ctx.fillStyle = 'rgba(103,174,102,.15)'; ctx.fillRect(0,0,w*.17,h*.58); ctx.fillRect(w*.84,0,w*.16,h*.56)
      ctx.fillStyle = 'rgba(242,155,70,.18)'; ctx.fillRect(0,h*.22,w*.13,h*.13)
      ctx.strokeStyle = 'rgba(177,132,210,.92)'; ctx.lineWidth = 2; ctx.setLineDash([6,5]); ctx.beginPath(); ctx.moveTo(w*.17,h*.56); ctx.quadraticCurveTo(w*.5,h*.39,w*.84,h*.56); ctx.stroke(); ctx.setLineDash([])
    }
    draw(); const observer = new ResizeObserver(draw); observer.observe(canvas); return () => observer.disconnect()
  }, [src, overlays])
  return <div className="image-stage"><img src={src} alt={alt} /><canvas ref={canvasRef} aria-hidden="true" />{overlays && <span className="overlay-caption">Illustrative regions · demo overlay</span>}</div>
}
