import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, ArrowDown, ArrowRight, ArrowUpRight, Camera, Check, ChevronDown, CircleHelp, CloudSun, Droplets, FileImage, Leaf, Menu, ShieldCheck, Sparkles, Upload, Waves, X } from 'lucide-react'
import { AssessmentCards, HumanDecisionPanel } from './components/AssessmentCards'
import ImageViewer from './components/ImageViewer'
import AuditLogCard from './components/AuditLogCard'
import { explainImage, initializeVision, inspectImage } from './ai'
import { requiresHumanReview } from './ai/validation'
import type { ImageQuality } from './ai/imageQuality'
import type { AssessmentRecord, Recommendations } from './types'

const DEMO='/demo-stream.svg'
const validTypes=['image/jpeg','image/png','image/webp']
const steps=['Upload photo','Review evidence','Confirm assessment']
const stored = ():AssessmentRecord[]=>{try{return JSON.parse(localStorage.getItem('rivervision.records')||'[]')}catch{return[]}}

function App() {
  const [src,setSrc]=useState(DEMO),[fileName,setFileName]=useState('Willow Creek · field sample'),[quality,setQuality]=useState<ImageQuality|null>(null)
  const [analysis,setAnalysis]=useState<Recommendations|null>(null),[answers,setAnswers]=useState<Record<string,string>>({}),[records,setRecords]=useState<AssessmentRecord[]>(stored)
  const [busy,setBusy]=useState(false),[aiReady,setAiReady]=useState<boolean|null>(null),[overlay,setOverlay]=useState(true),[drag,setDrag]=useState(false),[error,setError]=useState(''),[menuOpen,setMenuOpen]=useState(false)
  const inputRef=useRef<HTMLInputElement>(null)
  useEffect(()=>{initializeVision().then(setAiReady)},[])
  useEffect(()=>{return()=>{if(src.startsWith('blob:'))URL.revokeObjectURL(src)}},[src])
  const insufficient=quality!==null&&!quality.accepted
  const allAnswered=Boolean(answers.channel&&answers.water&&answers.impervious)
  const activeStep=records.length?3:analysis?2:1
  const confidence=analysis?[analysis.channel.confidence,analysis.water.confidence,analysis.impervious.confidence]:[]
  const reviewNote=useMemo(()=>analysis&&confidence.some((c,i)=>requiresHumanReview(c,[analysis.channel.sufficientEvidence,analysis.water.sufficientEvidence,analysis.impervious.sufficientEvidence][i])),[analysis])

  async function chooseFile(file?:File) {
    if(!file)return
    setError(''); if(!validTypes.includes(file.type)){setError('Please choose a JPG, PNG or WebP image.');return}
    if(src.startsWith('blob:'))URL.revokeObjectURL(src)
    const next=URL.createObjectURL(file);setSrc(next);setFileName(file.name);setAnalysis(null);setAnswers({});setQuality(null);setBusy(true)
    try { const q=await inspectImage(file);setQuality(q);if(!q.accepted){setError(q.message);return} const seed=Array.from(file.name).reduce((s,c)=>s+c.charCodeAt(0),0)+file.size; setAnalysis(explainImage(seed)) }
    catch {setError('This photo could not be read. Try exporting it as JPG, PNG or WebP.')} finally {setBusy(false)}
  }
  function finalize() {
    if(!analysis||!allAnswered)return
    const aiPrediction=analysis, humanDecision=answers
    const disagree=([['channel',analysis.channel.label],['water',analysis.water.label],['impervious',analysis.impervious.prediction]] as [string,string][]).some(([k,v])=>humanDecision[k]!==v)
    const record:AssessmentRecord={id:crypto.randomUUID(),fileName,timestamp:new Date().toISOString(),aiPrediction,humanDecision,finalDecision:{...humanDecision},disagreement:disagree}
    const next=[record,...records];setRecords(next);localStorage.setItem('rivervision.records',JSON.stringify(next))
    document.querySelector('#field-records')?.scrollIntoView({behavior:'smooth',block:'start'})
  }
  const reset=()=>{setSrc(DEMO);setFileName('Willow Creek · field sample');setQuality(null);setAnalysis(explainImage(2));setAnswers({});setError('')}
  useEffect(()=>{setAnalysis(explainImage(2))},[])

  return <div className="app-shell">
    <header className="topbar"><a href="#home" className="brand" aria-label="RiverVision home"><span className="brand-mark"><Waves size={21}/></span><span>river<span>vision</span></span></a><nav className="main-nav" aria-label="Main navigation"><a href="#how-it-works">How it works</a><a href="#assessment">Assessment</a><a href="#field-records">Field records</a></nav><div className="topbar-right"><span className="offline-status"><span/>On-device demo</span><button className="icon-button mobile-menu" aria-label={menuOpen?'Close menu':'Open menu'} aria-expanded={menuOpen} aria-controls="mobile-nav" onClick={()=>setMenuOpen(!menuOpen)}>{menuOpen?<X size={19}/>:<Menu size={19}/>}</button><a className="nav-cta" href="#assessment" onClick={()=>setMenuOpen(false)}>Start observing <ArrowUpRight size={15}/></a></div>{menuOpen&&<nav className="mobile-nav" id="mobile-nav" aria-label="Mobile navigation">{[['How it works','#how-it-works'],['Assessment','#assessment'],['Field records','#field-records']].map(([label,href])=><a key={href} href={href} onClick={()=>setMenuOpen(false)}>{label}<ArrowRight size={14}/></a>)}</nav>}</header>

    <main id="home">
      <section className="hero">
        <div className="hero-copy"><div className="eyebrow"><span className="eyebrow-line"/> CITIZEN SCIENCE, FLOWING FORWARD</div><h1>See your stream<br/>a little <em>differently.</em></h1><p className="hero-lede">A second set of eyes for the waterway you care about. Computer vision surfaces clues; your field knowledge makes the call.</p><div className="hero-actions"><a href="#assessment" className="primary-button">Start an assessment <ArrowRight size={17}/></a><a href="#how-it-works" className="text-link">How it works <ArrowDown size={14}/></a></div><div className="hero-footnote"><ShieldCheck size={15}/> Your photo stays on this device in this demo</div></div>
        <div className="hero-visual"><img className="hero-landscape" src={DEMO} alt="A winding stream flowing between green riverbanks"/><div className="hero-gradient"/><div className="hero-image-note"><span className="note-pulse"/><span>FIELD NOTE <strong>01 / 08</strong></span><span className="note-line"/><span>WILLOW CREEK</span></div><div className="floating-chip"><span className="chip-icon"><Droplets size={17}/></span><span><strong>A closer look</strong><small>Starts with your observation</small></span><ArrowUpRight size={15}/></div><div className="image-index">41° 52′ N&nbsp;&nbsp; · &nbsp;&nbsp; RIVER SYSTEMS</div></div>
      </section>

      <section className="trust-strip" aria-label="Responsible AI commitments">{[{icon:<CloudSun size={17}/>,title:'Browser-based AI',text:'Analysis on your device'},{icon:<EyeIcon/>,title:'Explainable by design',text:'See what the AI noticed'},{icon:<ShieldCheck size={17}/>,title:'Human verified',text:'You make the final call'},{icon:<Leaf size={17}/>,title:'For citizen science',text:'Field notes that add up'}].map(x=><div className="trust-item" key={x.title}><span className="trust-icon">{x.icon}</span><span><strong>{x.title}</strong><small>{x.text}</small></span></div>)}</section>

      <section className="assessment-section" id="assessment">
        <div className="section-heading"><div><div className="eyebrow"><span className="eyebrow-line"/> YOUR FIELD NOTE</div><h2>Make an observation</h2><p>Bring a photo from the water’s edge. We’ll help you look closer.</p></div><button className="help-button"><CircleHelp size={16}/> Field guide <ChevronDown size={14}/></button></div>
        <div className="workflow" aria-label={`Step ${activeStep} of 3`}>{steps.map((step,i)=><div key={step} className={`workflow-step ${i+1<=activeStep?'done':''} ${i+1===activeStep?'current':''}`}><span className="step-dot">{i+1<activeStep?<Check size={12}/>:`0${i+1}`}</span><span>{step}</span>{i<2&&<span className="step-track"/>}</div>)}</div>
        <div className="workspace-grid">
          <div className="photo-column"><section className="upload-card"><div className="card-heading"><div><span className="micro-label">01 — IMAGE</span><h3>Your stream, in frame</h3></div><span className="local-pill"><ShieldCheck size={13}/> Stays local</span></div>
            <div onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);void chooseFile(e.dataTransfer.files[0])}} className={`drop-zone ${drag?'dragging':''}`}>
              <ImageViewer src={src} alt="Preview of river field photo" overlays={overlay&&Boolean(analysis)}/>
              <div className="image-toolbar"><span><FileImage size={13}/>{fileName}</span><button onClick={()=>setOverlay(!overlay)} aria-pressed={overlay} className={overlay?'toolbar-active':''}><Sparkles size={13}/> Evidence overlay</button></div>
            </div>
            <div className="upload-actions"><button className="upload-button" onClick={()=>inputRef.current?.click()}><Upload size={16}/> Upload a photo</button><input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp" hidden onChange={e=>void chooseFile(e.target.files?.[0])}/><span>JPG, PNG or WebP · min. 640 × 360</span></div>
            {(quality||error||busy)&&<div className={`validation-panel ${error?'warning':''}`} role="status">{busy?<><span className="spinner"/>Reading image on this device…</>:error?<><span className="validation-icon">!</span><span>{error}</span></>:quality?<><span className="validation-icon">✓</span><span>{quality.message} <small>{quality.width} × {quality.height}px</small></span></>:null}</div>}
            <div className={`model-status ${aiReady===false?'fallback':''}`}><span className="status-dot"/><span>{aiReady===null?'Preparing browser vision…':aiReady?'TensorFlow.js · WebGL ready':'WebGL unavailable · demo mode'}</span><span className="demo-tag">DEMO MODEL</span></div>
          </section>
          <aside className="field-tip"><span className="tip-icon"><Activity size={15}/></span><p><strong>A small field tip</strong>Take your photo from the bank with the channel edges and the left margin in view. A single image can’t tell the whole story.</p></aside>
          <div className="disclaimer"><ShieldCheck size={15}/><span>Suggestions are illustrative demo outputs, not trained environmental classifiers. Visual clues are not measurements of water quality.</span></div>
          <div className="legend-card"><div className="legend-head"><span className="micro-label">IMAGE KEY</span><span>Illustrative overlay</span></div><div className="legend-items"><span><i className="legend-water"/>Water</span><span><i className="legend-veg"/>Vegetation</span><span><i className="legend-hard"/>Hard surfaces</span><span><i className="legend-bank"/>Bank edges</span></div></div>
          </div>
          <div className="review-column"><div className="review-heading"><div><span className="micro-label">02 — LOOK CLOSER</span><h3>What do you notice?</h3><p>Suggestions can be uncertain. Your observation is what counts.</p></div><span className="review-stamp"><Sparkles size={13}/> AI assisted</span></div>
            {!analysis?<div className="empty-analysis"><Camera size={22}/><strong>Your evidence is waiting</strong><span>Upload a photo to bring the field guide to life.</span></div>:<><div className="review-note"><Sparkles size={14}/><span>{reviewNote?'Some suggestions have low confidence. Please use your judgment.':'Suggestions are ready to review. Compare each one with what you can see.'}</span><span className="human-required">HUMAN REVIEW REQUIRED</span></div><AssessmentCards recommendations={analysis} answers={answers} setAnswers={setAnswers}/><HumanDecisionPanel complete={allAnswered} incomplete={insufficient||busy} onFinalize={finalize}/></>}
          </div>
        </div>
      </section>

      <section className="how-section" id="how-it-works"><div className="how-title"><div className="eyebrow"><span className="eyebrow-line"/> A BETTER KIND OF FIELD TOOL</div><h2>Technology brings the clues.<br/><em>You bring the context.</em></h2><p>River conditions are nuanced. This tool is built to support your attention, not replace the knowledge you gain by being there.</p></div><div className="how-cards">{[{n:'01',icon:<Camera/>,title:'Bring a photo',body:'Choose a clear view of your stream. Your photo is checked and stays in your browser.'},{n:'02',icon:<Sparkles/>,title:'Look for clues',body:'An illustrative AI demo highlights possible features and shares its reasoning.'},{n:'03',icon:<UserCheck/>,title:'Make the call',body:'Review each prompt, record what you observed, and keep the final decision yours.'}].map(item=><motion.article className="how-card" key={item.n} initial={{opacity:0,y:15}} whileInView={{opacity:1,y:0}} viewport={{once:true}} transition={{duration:.45,delay:Number(item.n)*.07}}><div className="how-card-top"><span className="how-icon">{item.icon}</span><span>{item.n}</span></div><h3>{item.title}</h3><p>{item.body}</p><span className="how-card-line"/></motion.article>)}</div></section>

      <section className="records-section" id="field-records"><div className="section-heading records-heading"><div><div className="eyebrow"><span className="eyebrow-line"/> YOUR FIELD NOTES</div><h2>Observations, kept close.</h2><p>Your confirmed assessments, stored only in this browser.</p></div><span className="record-count">{records.length} {records.length===1?'record':'records'}</span></div>
        {records.length?<div className="record-list">{records.map(r=><AuditLogCard key={r.id} record={r}/>)}</div>:<div className="records-empty"><span className="empty-leaf"><Waves size={21}/></span><div><strong>Your field notebook is ready.</strong><span>Complete a human-reviewed assessment and it will be kept here, on this device.</span></div><a href="#assessment">Start your first note <ArrowRight size={14}/></a></div>}
      </section>
    </main>
    <footer><a href="#home" className="brand footer-brand"><span className="brand-mark"><Waves size={18}/></span><span>river<span>vision</span></span></a><span>Thoughtful tools for waters worth knowing.</span><span className="footer-right">Made for people who care about their waterways <Leaf size={14}/></span></footer>
  </div>
}

function EyeIcon(){return <Activity size={17}/>}
function UserCheck(){return <ShieldCheck size={19}/>}
export default App
