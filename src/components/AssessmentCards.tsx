import { AlertCircle, Check, ChevronRight, Eye, Info, Sparkles, UserRound } from 'lucide-react'
import type { Recommendations } from '../types'
import { confidencePercent } from '../ai/confidence'

const channelChoices = [{ value:'flat',label:'Flat' },{ value:'u',label:'U shape' },{ value:'v',label:'V shape' }]
const waterChoices = [{ value:'clear',label:'Clear / transparent' },{ value:'muddy',label:'Muddy / turbid' },{ value:'foam',label:'Has foam' },{ value:'colored',label:'Has altered color' }]
const yesNo = [{value:'yes',label:'Yes'},{value:'no',label:'No'}]
type QuestionProps = { n:string; title:string; prompt:string; ai:string; confidence:number; reason:string; alternative:string; choices:{value:string;label:string}[]; answer?:string; onAnswer:(v:string)=>void; evidence:string; }
function Question({n,title,prompt,ai,confidence,reason,alternative,choices,answer,onAnswer,evidence}:QuestionProps) {
  return <article className="question-card">
    <div className="question-title"><span className="question-number">{n}</span><div><h3>{title}</h3><p>{prompt}</p></div></div>
    <div className="ai-suggestion"><div className="suggestion-top"><span className="ai-tag"><Sparkles size={13}/> AI suggestion</span><span className="confidence-label">{confidencePercent(confidence)} confidence</span></div><div className="meter"><span style={{width:confidencePercent(confidence)}}/></div><strong>{ai}</strong><div className="evidence-line"><Eye size={14}/><span>{evidence}</span></div><div className="explanation"><Info size={14}/><span>{reason} <em>Alternative: {alternative}</em></span></div></div>
    <div className="human-prompt"><div className="human-label"><UserRound size={15}/><span>Your observation</span><span className="required">Required</span></div><div className={`choice-list ${choices.length===4?'four':''}`} role="group" aria-label={`Your answer: ${title}`}>{choices.map(choice=><button type="button" key={choice.value} onClick={()=>onAnswer(choice.value)} aria-pressed={answer===choice.value} className={answer===choice.value?'selected':''}>{answer===choice.value&&<Check size={14}/>} {choice.label}</button>)}</div>{answer&&<span className="saved-note"><Check size={12}/> Your observation recorded</span>}</div>
  </article>
}
export function AssessmentCards({ recommendations, answers, setAnswers }: { recommendations:Recommendations; answers:Record<string,string>; setAnswers:(a:Record<string,string>)=>void }) {
  const set=(key:string,value:string)=>setAnswers({...answers,[key]:value})
  const c=recommendations.channel,w=recommendations.water,i=recommendations.impervious
  return <div className="questions-grid">
    <Question n="01" title="Channel form" prompt="What best describes the channel?" ai={`${c.label === 'u'?'U':c.label === 'v'?'V':'Flat'} shape`} confidence={c.confidence} reason={c.reason} alternative={c.alternative} choices={channelChoices} answer={answers.channel} onAnswer={v=>set('channel',v)} evidence="Channel banks" />
    <Question n="02" title="Water aspect" prompt="How is the water?" ai={{clear:'Clear / transparent',muddy:'Muddy / turbid',foam:'Has foam',colored:'Has altered color'}[w.label]} confidence={w.confidence} reason={w.reason} alternative={w.alternative} choices={waterChoices} answer={answers.water} onAnswer={v=>set('water',v)} evidence="Visible water surface" />
    <Question n="03" title="Impervious areas · left" prompt="Is more than one third of the left margin covered by roads, buildings, sidewalks or other impervious surfaces?" ai={i.prediction==='yes'?'Possibly present':'Not clearly visible'} confidence={i.confidence} reason={i.reason} alternative={i.alternative} choices={yesNo} answer={answers.impervious} onAnswer={v=>set('impervious',v)} evidence={i.detectedObjects.length?i.detectedObjects.join(', '):'No obvious hard surfaces'} />
  </div>
}
export function HumanDecisionPanel({ complete, onFinalize, incomplete }: { complete:boolean; onFinalize:()=>void; incomplete:boolean }) {
  return <div className="review-banner"><div className="review-icon"><AlertCircle size={20}/></div><div className="review-copy"><strong>Your judgment completes this assessment</strong><span>AI observations stay provisional. Review the image and answer all three questions to create your field record.</span></div><button type="button" disabled={!complete||incomplete} onClick={onFinalize}>Confirm assessment <ChevronRight size={16}/></button></div>
}
