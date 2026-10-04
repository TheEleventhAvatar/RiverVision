import { ArrowRight, ClipboardCheck, Clock3 } from 'lucide-react'
import type { AssessmentRecord } from '../types'
export default function AuditLogCard({record}:{record:AssessmentRecord}) {
  return <article className="audit-card"><header><span className="section-icon"><ClipboardCheck size={17}/></span><div><h3>Assessment record</h3><p>Human-confirmed · saved on this device</p></div><span className="record-badge">Complete</span></header>
    <div className="audit-meta"><span>{record.fileName}</span><span><Clock3 size={12}/> {new Date(record.timestamp).toLocaleString()}</span></div>
    {(['channel','water','impervious'] as const).map(key=>{const labels:Record<string,string>={flat:'Flat',u:'U shape',v:'V shape',clear:'Clear / transparent',muddy:'Muddy / turbid',foam:'Has foam',colored:'Altered color',yes:'Yes',no:'No'};const item=record.aiPrediction[key];const ai='label' in item?item.label:item.prediction;const human=record.humanDecision[key];return <div className="audit-row" key={key}><span className="audit-topic">{key==='impervious'?'Left margin':key==='channel'?'Channel form':'Water aspect'}</span><span className="audit-value"><small>AI</small>{labels[ai]??ai} <span className="audit-confidence">{Math.round(item.confidence*100)}%</span></span><ArrowRight size={15}/><span className="audit-value human"><small>Human</small>{labels[human]??human}</span>{record.disagreement&&ai!==human&&<span className="disagreement">Difference recorded</span>}</div>})}
  </article>
}
