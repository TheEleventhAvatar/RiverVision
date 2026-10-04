import{a as Ru}from"./index-zP4-j_0c.js";function Eu(n,e){for(var t=0;t<e.length;t++){const s=e[t];if(typeof s!="string"&&!Array.isArray(s)){for(const o in s)if(o!=="default"&&!(o in n)){const r=Object.getOwnPropertyDescriptor(s,o);r&&Object.defineProperty(n,o,r.get?r:{enumerable:!0,get:()=>s[o]})}}}return Object.freeze(Object.defineProperty(n,Symbol.toStringTag,{value:"Module"}))}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ku=1e-7,Nu=1e-4;class Au{constructor(e,t){this.backend=e,this.dataMover=t,this.data=new WeakMap,this.dataIdsCount=0}get(e){return this.data.has(e)||this.dataMover.moveData(this.backend,e),this.data.get(e)}set(e,t){this.dataIdsCount++,this.data.set(e,t)}has(e){return this.data.has(e)}delete(e){return this.dataIdsCount--,this.data.delete(e)}numDataIds(){return this.dataIdsCount}}class ga{refCount(e){return Ce("refCount")}incRef(e){return Ce("incRef")}timerAvailable(){return!0}time(e){return Ce("time")}read(e){return Ce("read")}readSync(e){return Ce("readSync")}readToGPU(e,t){return Ce("readToGPU")}numDataIds(){return Ce("numDataIds")}disposeData(e,t){return Ce("disposeData")}write(e,t,s){return Ce("write")}move(e,t,s,o,r){return Ce("move")}createTensorFromGPUData(e,t,s){return Ce("createTensorFromGPUData")}memory(){return Ce("memory")}floatPrecision(){return Ce("floatPrecision")}epsilon(){return this.floatPrecision()===32?ku:Nu}dispose(){return Ce("dispose")}}function Ce(n){throw new Error(`'${n}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xa(n){let e=n.length,t=0;for(;e>0;)t=Math.random()*e|0,e--,je(n,e,t)}function Fu(n,e){if(n.length!==e.length)throw new Error(`Array sizes must match to be shuffled together First array length was ${n.length}Second array length was ${e.length}`);let t=n.length,s=0;for(;t>0;)s=Math.random()*t|0,t--,je(n,t,s),je(e,t,s)}function wn(n,e,t){return Math.max(n,Math.min(e,t))}function ps(n){return n%2===0?n:n+1}function je(n,e,t){const s=n[e];n[e]=n[t],n[t]=s}function ba(n){let e=0;for(let t=0;t<n.length;t++)e+=n[t];return e}function Du(n,e){const t=Math.random();return e*t+(1-t)*n}function Ou(n,e){let t=0;for(let s=0;s<n.length;s++){const o=Number(n[s])-Number(e[s]);t+=o*o}return t}function R(n,e){if(!n)throw new Error(typeof e=="string"?e:e())}function Kt(n,e,t=""){R(oe(n,e),()=>t+` Shapes ${n} and ${e} must match`)}function Pu(n){R(n!=null,()=>"The input to the tensor constructor must be a non-null value.")}function A(n){if(n.length===0)return 1;let e=n[0];for(let t=1;t<n.length;t++)e*=n[t];return e}function _u(n){return n.length===0}function Lu(n,e){if(n===e)return!0;if(n==null||e==null||n.length!==e.length)return!1;for(let t=0;t<n.length;t++)if(n[t]!==null&&e[t]!==null&&n[t]!==e[t])return!1;return!0}function oe(n,e){if(n===e)return!0;if(n==null||e==null||n.length!==e.length)return!1;for(let t=0;t<n.length;t++)if(n[t]!==e[t])return!1;return!0}function Cn(n){return n%1===0}function Bu(n){if(Math.tanh!=null)return Math.tanh(n);if(n===1/0)return 1;if(n===-1/0)return-1;{const e=Math.exp(2*n);return(e-1)/(e+1)}}function rs(n){const e=Math.ceil(Math.sqrt(n));return[e,Math.ceil(n/e)]}function Mu(n){const e=new Uint32Array(n);for(let t=0;t<n;++t)e[t]=t;return xa(e),e}function bt(n,e){return e<=n.length?n:n+" ".repeat(e-n.length)}function Us(n,e=o=>0,t,s){return new Promise((o,r)=>{let a=0;const i=()=>{if(n()){o();return}a++;const c=e(a);if(t!=null&&a>=t){r();return}s!=null?s(i,c):setTimeout(i,c)};i()})}function wa(n,e){let t=1,s=-1;for(let r=0;r<n.length;++r)if(n[r]>=0)t*=n[r];else if(n[r]===-1){if(s!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${s} and dim ${r}`);s=r}else if(n[r]<0)throw Error(`Shapes can not be < 0. Found ${n[r]} at dim ${r}`);if(s===-1){if(e>0&&e!==t)throw Error(`Size(${e}) must match the product of shape ${n}`);return n}if(t===0)throw Error(`Cannot infer the missing size in [${n}] when there are 0 elements`);if(e%t!==0)throw Error(`The implicit shape can't be a fractional number. Got ${e} / ${t}`);const o=n.slice();return o[s]=e/t,o}function ce(n,e){const t=e.length;return n=n==null?e.map((s,o)=>o):[].concat(n),R(n.every(s=>s>=-t&&s<t),()=>`All values in axis param must be in range [-${t}, ${t}) but got axis ${n}`),R(n.every(s=>Cn(s)),()=>`All values in axis param must be integers but got axis ${n}`),n.map(s=>s<0?t+s:s)}function lt(n,e){const t=[],s=[],o=e!=null&&Array.isArray(e)&&e.length===0,r=e==null||o?null:ce(e,n).sort();let a=0;for(let i=0;i<n.length;++i){if(r!=null){if(r[a]===i&&n[i]!==1)throw new Error(`Can't squeeze axis ${i} since its dim '${n[i]}' is not 1`);(r[a]==null||r[a]>i)&&n[i]===1&&(t.push(n[i]),s.push(i)),r[a]<=i&&a++}n[i]!==1&&(t.push(n[i]),s.push(i))}return{newShape:t,keptDims:s}}function Ue(n,e){return te(n,e)}function te(n,e){let t=null;if(n==null||n==="float32")t=new Float32Array(e);else if(n==="int32")t=new Int32Array(e);else if(n==="bool")t=new Uint8Array(e);else if(n==="string")t=new Array(e);else throw new Error(`Unknown data type ${n}`);return t}function Ca(n,e){for(let t=0;t<n.length;t++){const s=n[t];if(isNaN(s)||!isFinite(s))throw Error(`A tensor of type ${e} being uploaded contains ${s}.`)}}function ya(n){return n==="bool"||n==="complex64"||n==="float32"||n==="int32"||n==="string"}function po(n,e){return!(e==="complex64"||e==="float32"&&n!=="complex64"||e==="int32"&&n!=="float32"&&n!=="complex64"||e==="bool"&&n==="bool")}function yn(n){if(n==="float32"||n==="int32")return 4;if(n==="complex64")return 8;if(n==="bool")return 1;throw new Error(`Unknown dtype ${n}`)}function $a(n){if(n==null)return 0;let e=0;return n.forEach(t=>e+=t.length),e}function Fn(n){return typeof n=="string"||n instanceof String}function va(n){return typeof n=="boolean"}function Sa(n){return typeof n=="number"}function Yt(n){return Array.isArray(n)?Yt(n[0]):n instanceof Float32Array?"float32":n instanceof Int32Array||n instanceof Uint8Array||n instanceof Uint8ClampedArray?"int32":Sa(n)?"float32":Fn(n)?"string":va(n)?"bool":"float32"}function rt(n){return!!(n&&n.constructor&&n.call&&n.apply)}function as(n,e){for(let t=e;t<n;++t)if(n%t===0)return t;return n}function J(n){const e=n.length;if(e<2)return[];const t=new Array(e-1);t[e-2]=n[e-1];for(let s=e-3;s>=0;--s)t[s]=t[s+1]*n[s+1];return t}function Ia(n,e,t,s=!1){const o=new Array;if(e.length===1){const r=e[0]*(s?2:1);for(let a=0;a<r;a++)o[a]=t[n+a]}else{const r=e[0],a=e.slice(1),i=a.reduce((c,l)=>c*l)*(s?2:1);for(let c=0;c<r;c++)o[c]=Ia(n+c*i,a,t,s)}return o}function Ut(n,e,t=!1){if(n.length===0)return e[0];const s=n.reduce((o,r)=>o*r)*(t?2:1);if(s===0)return[];if(s!==e.length)throw new Error(`[${n}] does not match the input size ${e.length}${t?" for a complex tensor":""}.`);return Ia(0,n,e,t)}function Vu(n,e){if(Array.isArray(n))return n;if(e==="float32")return n instanceof Float32Array?n:new Float32Array(n);if(e==="int32")return n instanceof Int32Array?n:new Int32Array(n);if(e==="bool"||e==="string")return Uint8Array.from(new Int32Array(n));throw new Error(`Unknown dtype ${e}`)}function Ta(n,e){const t=We(n,e);for(let s=0;s<t.length;s++)t[s]=1;return t}function We(n,e){if(e==null||e==="float32"||e==="complex64")return new Float32Array(n);if(e==="int32")return new Int32Array(n);if(e==="bool")return new Uint8Array(n);throw new Error(`Unknown data type ${e}`)}function Uu(n,e){const t=n.reduce((s,o)=>s*o,1);if(e==null||e==="float32")return Ut(n,new Float32Array(t));if(e==="int32")return Ut(n,new Int32Array(t));if(e==="bool")return Ut(n,new Uint8Array(t));throw new Error(`Unknown data type ${e}`)}function Qt(n){n.forEach(e=>{R(Number.isInteger(e)&&e>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${n}].`)})}function Ht(n,e,t){if(e===0)return 0;if(e===1)return n[0];let s=n[n.length-1];for(let o=0;o<n.length-1;++o)s+=t[o]*n[o];return s}function Dn(n,e,t){if(e===0)return[];if(e===1)return[n];const s=new Array(e);for(let o=0;o<s.length-1;++o)s[o]=Math.floor(n/t[o]),n-=s[o]*t[o];return s[s.length-1]=n,s}function ms(n){return n&&n.then&&typeof n.then=="function"}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yr="tfjsflags";class Wu{constructor(e){this.global=e,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=Gu,this.populateURLFlags()}setPlatform(e,t){this.platform!=null&&($().getBool("IS_TEST")||$().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${e}.`)),this.platformName=e,this.platform=t}registerFlag(e,t,s){if(this.flagRegistry[e]={evaluationFn:t,setHook:s},this.urlFlags[e]!=null){const o=this.urlFlags[e];$().getBool("IS_TEST")||$().getBool("PROD")||console.warn(`Setting feature override from URL ${e}: ${o}.`),this.set(e,o)}}async getAsync(e){return e in this.flags?this.flags[e]:(this.flags[e]=await this.evaluateFlag(e),this.flags[e])}get(e){if(e in this.flags)return this.flags[e];const t=this.evaluateFlag(e);if(ms(t))throw new Error(`Flag ${e} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[e]=t,this.flags[e]}getNumber(e){return this.get(e)}getBool(e){return this.get(e)}getString(e){return this.get(e)}getFlags(){return this.flags}get features(){return this.flags}set(e,t){if(this.flagRegistry[e]==null)throw new Error(`Cannot set flag ${e} as it has not been registered.`);this.flags[e]=t,this.flagRegistry[e].setHook!=null&&this.flagRegistry[e].setHook(t)}evaluateFlag(e){if(this.flagRegistry[e]==null)throw new Error(`Cannot evaluate flag '${e}': no evaluation function found.`);return this.flagRegistry[e].evaluationFn()}setFlags(e){this.flags=Object.assign({},e)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const e=this.getQueryParams(this.global.location.search);yr in e&&e[yr].split(",").forEach(s=>{const[o,r]=s.split(":");this.urlFlags[o]=Hu(o,r)})}}function Gu(n){const e={};return n.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(t,...s)=>(zu(e,s[0],s[1]),s.join("="))),e}function zu(n,e,t){n[decodeURIComponent(e)]=decodeURIComponent(t||"")}function Hu(n,e){const t=e.toLowerCase();return t==="true"||t==="false"?t==="true":`${+t}`===t?+t:e}function $(){return Ra}let Ra=null;function Xu(n){Ra=n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Ds;function Ea(){if(Ds==null){let n;if(typeof window<"u")n=window;else if(typeof global<"u")n=global;else if(typeof process<"u")n=process;else if(typeof self<"u")n=self;else throw new Error("Could not find a global object");Ds=n}return Ds}function qu(){const n=Ea();return n._tfGlobals==null&&(n._tfGlobals=new Map),n._tfGlobals}function mo(n,e){const t=qu();if(t.has(n))return t.get(n);{const s=e();return t.set(n,s),t.get(n)}}const go="Abs",ju="Acos",Ku="Acosh",On="Add",Yu="AddN",Qu="All",Zu="Any",Ju="ArgMax",ed="ArgMin",td="Asin",nd="Asinh",sd="Atan",od="Atanh",rd="Atan2",ad="AvgPool",id="AvgPoolGrad",cd="AvgPool3D",ld="AvgPool3DGrad",ud="BatchMatMul",dd="BatchToSpaceND",hd="Bincount",xo="BitwiseAnd",RT="BroadcastTo",fd="BroadcastArgs",gs="Cast",bo="Ceil",pd="ClipByValue",wo="Complex",ka="ComplexAbs",md="Concat",gd="Conv2D",xd="Conv2DBackpropFilter",bd="Conv2DBackpropInput",wd="Conv3D",Cd="Conv3DBackpropFilterV2",yd="Conv3DBackpropInputV2",$d="Cos",vd="Cosh",Sd="Cumprod",Id="Cumsum",Td="CropAndResize",Rd="DenseBincount",Ed="DepthToSpace",kd="DepthwiseConv2dNative",Nd="DepthwiseConv2dNativeBackpropFilter",Ad="DepthwiseConv2dNativeBackpropInput",Fd="Diag",Dd="Dilation2D",ET="Dilation2DBackpropInput",kT="Dilation2DBackpropFilter",NT="Draw",Na="RealDiv",Od="Einsum",Aa="Elu",Pd="EluGrad",_d="Erf",Co="Equal",yo="Exp",Ld="ExpandDims",$o="Expm1",Bd="FFT",Fa="Fill",Md="FlipLeftRight",vo="Floor",xs="FloorDiv",Vd="FusedBatchNorm",Ud="GatherV2",Wd="GatherNd",So="Greater",Io="GreaterEqual",bs="Identity",Gd="IFFT",zd="Imag",Hd="IsFinite",Xd="IsInf",qd="IsNan",Da="LeakyRelu",To="Less",Ro="LessEqual",jd="LinSpace",Eo="Log",Kd="Log1p",Yd="LogicalAnd",Qd="LogicalNot",Zd="LogicalOr",AT="LogicalXor",FT="LogSoftmax",DT="LowerBound",Jd="LRN",eh="LRNGrad",OT="MatrixBandPart",th="Max",ws="Maximum",nh="MaxPool",sh="MaxPoolGrad",oh="MaxPool3D",rh="MaxPool3DGrad",ah="MaxPoolWithArgmax",ih="Mean",ch="Min",ko="Minimum",lh="MirrorPad",uh="Mod",dh="Multinomial",Cs="Multiply",Oa="Neg",No="NotEqual",hh="NonMaxSuppressionV3",fh="NonMaxSuppressionV4",ph="NonMaxSuppressionV5",mh="OnesLike",gh="OneHot",xh="Pack",bh="PadV2",PT="Pool",Pa="Pow",_a="Prelu",La="Prod",wh="RaggedGather",Ch="RaggedRange",yh="RaggedTensorToTensor",$h="Range",Ba="Real",vh="Reciprocal",Ma="Relu",Va="Reshape",Sh="ResizeNearestNeighbor",Ih="ResizeNearestNeighborGrad",Th="ResizeBilinear",Rh="ResizeBilinearGrad",Ua="Relu6",Eh="Reverse",kh="Round",Ao="Rsqrt",Nh="ScatterNd",Ah="TensorScatterUpdate",Fh="SearchSorted",Dh="Select",Oh="Selu",Wa="Slice",Ph="Sin",_h="Sinh",Lh="Sign",ys="Sigmoid",Bh="Softplus",$s="Sqrt",Ga="Sum",Mh="SpaceToBatchND",Vh="SplitV",Uh="Softmax",Wh="SparseFillEmptyRows",Gh="SparseReshape",zh="SparseSegmentMean",Hh="SparseSegmentSum",Xh="SparseToDense",Fo="SquaredDifference",qh="Square",Do="StaticRegexReplace",jh="StridedSlice",Kh="StringNGrams",Yh="StringSplit",Qh="StringToHashBucketFast",vs="Sub",Zh="Tan",Jh="Tanh",za="Tile",ef="TopK",tf="Transform",Ha="Transpose",nf="Unique",sf="Unpack",of="UnsortedSegmentSum",_T="UpperBound",Xa="ZerosLike",qa="Step",rf="FromPixels",af="RotateWithOffset",cf="_FusedMatMul",lf="FusedConv2D",uf="FusedDepthwiseConv2D";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ae(...n){$().getBool("IS_TEST")||$().getBool("PROD")||console.warn(...n)}function df(...n){$().getBool("IS_TEST")||$().getBool("PROD")||console.log(...n)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xt=mo("kernelRegistry",()=>new Map),$n=mo("gradRegistry",()=>new Map);function $r(n,e){const t=Oo(n,e);return Xt.get(t)}function vr(n){return $n.get(n)}function Ws(n){const e=Xt.entries(),t=[];for(;;){const{done:s,value:o}=e.next();if(s)break;const[r,a]=o,[i]=r.split("_");i===n&&t.push(a)}return t}function ja(n){const{kernelName:e,backendName:t}=n,s=Oo(e,t);Xt.has(s)&&Ae(`The kernel '${e}' for backend '${t}' is already registered`),Xt.set(s,n)}function LT(n){const{kernelName:e}=n;$n.has(e)&&$().getBool("DEBUG")&&Ae(`Overriding the gradient for '${e}'`),$n.set(e,n)}function BT(n,e){const t=Oo(n,e);if(!Xt.has(t))throw new Error(`The kernel '${n}' for backend '${e}' is not registered`);Xt.delete(t)}function MT(n){if(!$n.has(n))throw new Error(`The gradient '${n}' for backend is not registered`);$n.delete(n)}function VT(n,e){Ws(n).forEach(s=>{const o=Object.assign({},s,{backendName:e});ja(o)})}function Oo(n,e){return`${e}_${n}`}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ka(n){return n instanceof Float32Array||n instanceof Int32Array||n instanceof Uint8Array||n instanceof Uint8ClampedArray}var Os,Sr;function hf(){if(Sr)return Os;Sr=1,Os=e;var n=null;try{n=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function e(v,w,E){this.low=v|0,this.high=w|0,this.unsigned=!!E}e.prototype.__isLong__,Object.defineProperty(e.prototype,"__isLong__",{value:!0});function t(v){return(v&&v.__isLong__)===!0}e.isLong=t;var s={},o={};function r(v,w){var E,_,B;return w?(v>>>=0,(B=0<=v&&v<256)&&(_=o[v],_)?_:(E=i(v,(v|0)<0?-1:0,!0),B&&(o[v]=E),E)):(v|=0,(B=-128<=v&&v<128)&&(_=s[v],_)?_:(E=i(v,v<0?-1:0,!1),B&&(s[v]=E),E))}e.fromInt=r;function a(v,w){if(isNaN(v))return w?b:m;if(w){if(v<0)return b;if(v>=p)return k}else{if(v<=-x)return T;if(v+1>=x)return N}return v<0?a(-v,w).neg():i(v%f|0,v/f|0,w)}e.fromNumber=a;function i(v,w,E){return new e(v,w,E)}e.fromBits=i;var c=Math.pow;function l(v,w,E){if(v.length===0)throw Error("empty string");if(v==="NaN"||v==="Infinity"||v==="+Infinity"||v==="-Infinity")return m;if(typeof w=="number"?(E=w,w=!1):w=!!w,E=E||10,E<2||36<E)throw RangeError("radix");var _;if((_=v.indexOf("-"))>0)throw Error("interior hyphen");if(_===0)return l(v.substring(1),w,E).neg();for(var B=a(c(E,8)),L=m,W=0;W<v.length;W+=8){var j=Math.min(8,v.length-W),G=parseInt(v.substring(W,W+j),E);if(j<8){var K=a(c(E,j));L=L.mul(K).add(a(G))}else L=L.mul(B),L=L.add(a(G))}return L.unsigned=w,L}e.fromString=l;function u(v,w){return typeof v=="number"?a(v,w):typeof v=="string"?l(v,w):i(v.low,v.high,typeof w=="boolean"?w:v.unsigned)}e.fromValue=u;var d=65536,h=1<<24,f=d*d,p=f*f,x=p/2,g=r(h),m=r(0);e.ZERO=m;var b=r(0,!0);e.UZERO=b;var C=r(1);e.ONE=C;var y=r(1,!0);e.UONE=y;var S=r(-1);e.NEG_ONE=S;var N=i(-1,2147483647,!1);e.MAX_VALUE=N;var k=i(-1,-1,!0);e.MAX_UNSIGNED_VALUE=k;var T=i(0,-2147483648,!1);e.MIN_VALUE=T;var I=e.prototype;return I.toInt=function(){return this.unsigned?this.low>>>0:this.low},I.toNumber=function(){return this.unsigned?(this.high>>>0)*f+(this.low>>>0):this.high*f+(this.low>>>0)},I.toString=function(w){if(w=w||10,w<2||36<w)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative())if(this.eq(T)){var E=a(w),_=this.div(E),B=_.mul(E).sub(this);return _.toString(w)+B.toInt().toString(w)}else return"-"+this.neg().toString(w);for(var L=a(c(w,6),this.unsigned),W=this,j="";;){var G=W.div(L),K=W.sub(G.mul(L)).toInt()>>>0,X=K.toString(w);if(W=G,W.isZero())return X+j;for(;X.length<6;)X="0"+X;j=""+X+j}},I.getHighBits=function(){return this.high},I.getHighBitsUnsigned=function(){return this.high>>>0},I.getLowBits=function(){return this.low},I.getLowBitsUnsigned=function(){return this.low>>>0},I.getNumBitsAbs=function(){if(this.isNegative())return this.eq(T)?64:this.neg().getNumBitsAbs();for(var w=this.high!=0?this.high:this.low,E=31;E>0&&(w&1<<E)==0;E--);return this.high!=0?E+33:E+1},I.isZero=function(){return this.high===0&&this.low===0},I.eqz=I.isZero,I.isNegative=function(){return!this.unsigned&&this.high<0},I.isPositive=function(){return this.unsigned||this.high>=0},I.isOdd=function(){return(this.low&1)===1},I.isEven=function(){return(this.low&1)===0},I.equals=function(w){return t(w)||(w=u(w)),this.unsigned!==w.unsigned&&this.high>>>31===1&&w.high>>>31===1?!1:this.high===w.high&&this.low===w.low},I.eq=I.equals,I.notEquals=function(w){return!this.eq(w)},I.neq=I.notEquals,I.ne=I.notEquals,I.lessThan=function(w){return this.comp(w)<0},I.lt=I.lessThan,I.lessThanOrEqual=function(w){return this.comp(w)<=0},I.lte=I.lessThanOrEqual,I.le=I.lessThanOrEqual,I.greaterThan=function(w){return this.comp(w)>0},I.gt=I.greaterThan,I.greaterThanOrEqual=function(w){return this.comp(w)>=0},I.gte=I.greaterThanOrEqual,I.ge=I.greaterThanOrEqual,I.compare=function(w){if(t(w)||(w=u(w)),this.eq(w))return 0;var E=this.isNegative(),_=w.isNegative();return E&&!_?-1:!E&&_?1:this.unsigned?w.high>>>0>this.high>>>0||w.high===this.high&&w.low>>>0>this.low>>>0?-1:1:this.sub(w).isNegative()?-1:1},I.comp=I.compare,I.negate=function(){return!this.unsigned&&this.eq(T)?T:this.not().add(C)},I.neg=I.negate,I.add=function(w){t(w)||(w=u(w));var E=this.high>>>16,_=this.high&65535,B=this.low>>>16,L=this.low&65535,W=w.high>>>16,j=w.high&65535,G=w.low>>>16,K=w.low&65535,X=0,ne=0,Z=0,ue=0;return ue+=L+K,Z+=ue>>>16,ue&=65535,Z+=B+G,ne+=Z>>>16,Z&=65535,ne+=_+j,X+=ne>>>16,ne&=65535,X+=E+W,X&=65535,i(Z<<16|ue,X<<16|ne,this.unsigned)},I.subtract=function(w){return t(w)||(w=u(w)),this.add(w.neg())},I.sub=I.subtract,I.multiply=function(w){if(this.isZero())return m;if(t(w)||(w=u(w)),n){var E=n.mul(this.low,this.high,w.low,w.high);return i(E,n.get_high(),this.unsigned)}if(w.isZero())return m;if(this.eq(T))return w.isOdd()?T:m;if(w.eq(T))return this.isOdd()?T:m;if(this.isNegative())return w.isNegative()?this.neg().mul(w.neg()):this.neg().mul(w).neg();if(w.isNegative())return this.mul(w.neg()).neg();if(this.lt(g)&&w.lt(g))return a(this.toNumber()*w.toNumber(),this.unsigned);var _=this.high>>>16,B=this.high&65535,L=this.low>>>16,W=this.low&65535,j=w.high>>>16,G=w.high&65535,K=w.low>>>16,X=w.low&65535,ne=0,Z=0,ue=0,Lt=0;return Lt+=W*X,ue+=Lt>>>16,Lt&=65535,ue+=L*X,Z+=ue>>>16,ue&=65535,ue+=W*K,Z+=ue>>>16,ue&=65535,Z+=B*X,ne+=Z>>>16,Z&=65535,Z+=L*K,ne+=Z>>>16,Z&=65535,Z+=W*G,ne+=Z>>>16,Z&=65535,ne+=_*X+B*K+L*G+W*j,ne&=65535,i(ue<<16|Lt,ne<<16|Z,this.unsigned)},I.mul=I.multiply,I.divide=function(w){if(t(w)||(w=u(w)),w.isZero())throw Error("division by zero");if(n){if(!this.unsigned&&this.high===-2147483648&&w.low===-1&&w.high===-1)return this;var E=(this.unsigned?n.div_u:n.div_s)(this.low,this.high,w.low,w.high);return i(E,n.get_high(),this.unsigned)}if(this.isZero())return this.unsigned?b:m;var _,B,L;if(this.unsigned){if(w.unsigned||(w=w.toUnsigned()),w.gt(this))return b;if(w.gt(this.shru(1)))return y;L=b}else{if(this.eq(T)){if(w.eq(C)||w.eq(S))return T;if(w.eq(T))return C;var W=this.shr(1);return _=W.div(w).shl(1),_.eq(m)?w.isNegative()?C:S:(B=this.sub(w.mul(_)),L=_.add(B.div(w)),L)}else if(w.eq(T))return this.unsigned?b:m;if(this.isNegative())return w.isNegative()?this.neg().div(w.neg()):this.neg().div(w).neg();if(w.isNegative())return this.div(w.neg()).neg();L=m}for(B=this;B.gte(w);){_=Math.max(1,Math.floor(B.toNumber()/w.toNumber()));for(var j=Math.ceil(Math.log(_)/Math.LN2),G=j<=48?1:c(2,j-48),K=a(_),X=K.mul(w);X.isNegative()||X.gt(B);)_-=G,K=a(_,this.unsigned),X=K.mul(w);K.isZero()&&(K=C),L=L.add(K),B=B.sub(X)}return L},I.div=I.divide,I.modulo=function(w){if(t(w)||(w=u(w)),n){var E=(this.unsigned?n.rem_u:n.rem_s)(this.low,this.high,w.low,w.high);return i(E,n.get_high(),this.unsigned)}return this.sub(this.div(w).mul(w))},I.mod=I.modulo,I.rem=I.modulo,I.not=function(){return i(~this.low,~this.high,this.unsigned)},I.and=function(w){return t(w)||(w=u(w)),i(this.low&w.low,this.high&w.high,this.unsigned)},I.or=function(w){return t(w)||(w=u(w)),i(this.low|w.low,this.high|w.high,this.unsigned)},I.xor=function(w){return t(w)||(w=u(w)),i(this.low^w.low,this.high^w.high,this.unsigned)},I.shiftLeft=function(w){return t(w)&&(w=w.toInt()),(w&=63)===0?this:w<32?i(this.low<<w,this.high<<w|this.low>>>32-w,this.unsigned):i(0,this.low<<w-32,this.unsigned)},I.shl=I.shiftLeft,I.shiftRight=function(w){return t(w)&&(w=w.toInt()),(w&=63)===0?this:w<32?i(this.low>>>w|this.high<<32-w,this.high>>w,this.unsigned):i(this.high>>w-32,this.high>=0?0:-1,this.unsigned)},I.shr=I.shiftRight,I.shiftRightUnsigned=function(w){if(t(w)&&(w=w.toInt()),w&=63,w===0)return this;var E=this.high;if(w<32){var _=this.low;return i(_>>>w|E<<32-w,E>>>w,this.unsigned)}else return w===32?i(E,0,this.unsigned):i(E>>>w-32,0,this.unsigned)},I.shru=I.shiftRightUnsigned,I.shr_u=I.shiftRightUnsigned,I.toSigned=function(){return this.unsigned?i(this.low,this.high,!1):this},I.toUnsigned=function(){return this.unsigned?this:i(this.low,this.high,!0)},I.toBytes=function(w){return w?this.toBytesLE():this.toBytesBE()},I.toBytesLE=function(){var w=this.high,E=this.low;return[E&255,E>>>8&255,E>>>16&255,E>>>24,w&255,w>>>8&255,w>>>16&255,w>>>24]},I.toBytesBE=function(){var w=this.high,E=this.low;return[w>>>24,w>>>16&255,w>>>8&255,w&255,E>>>24,E>>>16&255,E>>>8&255,E&255]},e.fromBytes=function(w,E,_){return _?e.fromBytesLE(w,E):e.fromBytesBE(w,E)},e.fromBytesLE=function(w,E){return new e(w[0]|w[1]<<8|w[2]<<16|w[3]<<24,w[4]|w[5]<<8|w[6]<<16|w[7]<<24,E)},e.fromBytesBE=function(w,E){return new e(w[4]<<24|w[5]<<16|w[6]<<8|w[7],w[0]<<24|w[1]<<16|w[2]<<8|w[3],E)},Os}var Ya=hf();const Qa=Ru(Ya),ff=Eu({__proto__:null,default:Qa},[Ya]);/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mt=Qa||ff;function Pn(n){return mt.fromString(n,!0,16)}const Za=Pn("c3a5c85c97cb3127"),pt=Pn("b492b66fbe98f273"),fe=Pn("9ae16a3b2f90404f");function Gs(n){return n.xor(n.shru(47))}function Ja(n,e,t){const s=n.slice(e,e+t);return mt.fromBytes(Array.from(s),!0,!0)}function q(n,e){return Ja(n,e,8)}function Ir(n,e){return Ja(n,e,4)}function se(n,e){return e===0?n:n.shru(e).or(n.shl(64-e))}function st(n,e,t=Pn("9ddfea08eb382d69")){let s=n.xor(e).mul(t);s=s.xor(s.shru(47));let o=e.xor(s).mul(t);return o=o.xor(o.shru(47)),o=o.mul(t),o}function pf(n,e,t,s,o,r){o=o.add(n),r=se(r.add(o).add(s),21);const a=o;return o=o.add(e),o=o.add(t),r=r.add(se(o,44)),[o.add(s),r.add(a)]}function Hn(n,e,t,s){return pf(q(n,e),q(n,e+8),q(n,e+16),q(n,e+24),t,s)}function mf(n,e=n.length){if(e>=8){const t=fe.add(e*2),s=q(n,0).add(fe),o=q(n,e-8),r=se(o,37).mul(t).add(s),a=se(s,25).add(o).mul(t);return st(r,a,t)}if(e>=4){const t=fe.add(e*2),s=Ir(n,0);return st(s.shl(3).add(e),Ir(n,e-4),t)}if(e>0){const t=n[0],s=n[e>>1],o=n[e-1],r=t+(s<<8),a=e+(o<<2);return Gs(fe.mul(r).xor(Za.mul(a))).mul(fe)}return fe}function gf(n,e=n.length){const t=fe.add(e*2),s=q(n,0).mul(pt),o=q(n,8),r=q(n,e-8).mul(t),a=q(n,e-16).mul(fe);return st(se(s.add(o),43).add(se(r,30)).add(a),s.add(se(o.add(fe),18)).add(r),t)}function xf(n,e=n.length){const t=fe.add(e*2),s=q(n,0).mul(fe),o=q(n,8),r=q(n,e-8).mul(t),a=q(n,e-16).mul(fe),i=se(s.add(o),43).add(se(r,30)).add(a),c=st(i,s.add(se(o.add(fe),18)).add(r),t),l=q(n,16).mul(t),u=q(n,24),d=i.add(q(n,e-32)).mul(t),h=c.add(q(n,e-24)).mul(t);return st(se(l.add(u),43).add(se(d,30)).add(h),l.add(se(u.add(s),18)).add(d),t)}function ei(n,e=n.length){const t=mt.fromNumber(81,!0);if(e<=32)return e<=16?mf(n,e):gf(n,e);if(e<=64)return xf(n,e);let s=t,o=t.mul(pt).add(113),r=Gs(o.mul(fe).add(113)).mul(fe),a=[mt.UZERO,mt.UZERO],i=[mt.UZERO,mt.UZERO];s=s.mul(fe).add(q(n,0));let c=0;const l=(e-1>>6)*64,u=l+(e-1&63)-63;do s=se(s.add(o).add(a[0]).add(q(n,c+8)),37).mul(pt),o=se(o.add(a[1]).add(q(n,c+48)),42).mul(pt),s=s.xor(i[1]),o=o.add(a[0]).add(q(n,c+40)),r=se(r.add(i[0]),33).mul(pt),a=Hn(n,c,a[1].mul(pt),s.add(i[0])),i=Hn(n,c+32,r.add(i[1]),o.add(q(n,c+16))),[r,s]=[s,r],c+=64;while(c!==l);const d=pt.add(r.and(255).shl(1));return c=u,i[0]=i[0].add(e-1&63),a[0]=a[0].add(i[0]),i[0]=i[0].add(a[0]),s=se(s.add(o).add(a[0]).add(q(n,c+8)),37).mul(d),o=se(o.add(a[1]).add(q(n,c+48)),42).mul(d),s=s.xor(i[1].mul(9)),o=o.add(a[0].mul(9).add(q(n,c+40))),r=se(r.add(i[0]),33).mul(d),a=Hn(n,c,a[1].mul(d),s.add(i[0])),i=Hn(n,c+32,r.add(i[1]),o.add(q(n,c+16))),[r,s]=[s,r],st(st(a[0],i[0],d).add(Gs(o).mul(Za)).add(r),st(a[1],i[1],d).add(s),d)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nt(n,e){return e==="string"?ot(n):_n([n],e)}function bf(n,e){return n instanceof Float32Array&&e==="float32"||n instanceof Int32Array&&e==="int32"||n instanceof Uint8Array&&e==="bool"}function _n(n,e){if(e==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(n)&&(n=at(n)),$().getBool("DEBUG")&&Ca(n,e),bf(n,e))return n;if(e==null||e==="float32"||e==="complex64")return new Float32Array(n);if(e==="int32")return new Int32Array(n);if(e==="bool"){const t=new Uint8Array(n.length);for(let s=0;s<t.length;++s)Math.round(n[s])!==0&&(t[s]=1);return t}else throw new Error(`Unknown data type ${e}`)}function Ne(){return $().platform.now()}function wf(n,e){return $().platform.fetch(n,e)}function ot(n,e="utf-8"){return e=e||"utf-8",$().platform.encode(n,e)}function Ct(n,e="utf-8"){return e=e||"utf-8",$().platform.decode(n,e)}function Te(n){return $().platform.isTypedArray!=null?$().platform.isTypedArray(n):Ka(n)}function at(n,e=[],t=!1){if(e==null&&(e=[]),typeof n=="boolean"||typeof n=="number"||typeof n=="string"||ms(n)||n==null||Te(n)&&t)e.push(n);else if(Array.isArray(n)||Te(n))for(let s=0;s<n.length;++s)at(n[s],e,t);else{let s=-1;for(const o of Object.keys(n))/^([1-9]+[0-9]*|0)$/.test(o)&&(s=Math.max(s,Number(o)));for(let o=0;o<=s;o++)at(n[o],e,t)}return e}const UT=Object.freeze(Object.defineProperty({__proto__:null,arraysEqual:oe,arraysEqualWithNull:Lu,assert:R,assertNonNegativeIntegerDimensions:Qt,assertNonNull:Pu,assertShapesMatch:Kt,bytesFromStringArray:$a,bytesPerElement:yn,checkConversionForErrors:Ca,clamp:wn,computeStrides:J,convertBackendValuesAndArrayBuffer:Vu,createScalarValue:Nt,createShuffledIndices:Mu,decodeString:Ct,distSquared:Ou,encodeString:ot,fetch:wf,fingerPrint64:ei,flatten:at,getArrayFromDType:te,getTypedArrayFromDType:Ue,hasEncodingLoss:po,hexToLong:Pn,indexToLoc:Dn,inferDtype:Yt,inferFromImplicitShape:wa,isBoolean:va,isFunction:rt,isInt:Cn,isNumber:Sa,isPromise:ms,isScalarShape:_u,isString:Fn,isTypedArray:Te,isValidDtype:ya,locToIndex:Ht,makeOnesTypedArray:Ta,makeZerosNestedTypedArray:Uu,makeZerosTypedArray:We,nearestDivisor:as,nearestLargerEven:ps,now:Ne,parseAxisParam:ce,randUniform:Du,repeatedTry:Us,rightPad:bt,shuffle:xa,shuffleCombo:Fu,sizeFromShape:A,sizeToSquarishShape:rs,squeezeShape:lt,sum:ba,swap:je,tanh:Bu,toNestedArray:Ut,toTypedArray:_n},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Cf{constructor(e,t){this.backendTimer=e,this.logger=t,t==null&&(this.logger=new $f)}profileKernel(e,t,s){let o;const r=()=>{o=s()};let a;const i=Ne();if(this.backendTimer.timerAvailable())a=this.backendTimer.time(r);else{r();for(const l of o)l.dataSync();a=Promise.resolve({kernelMs:Ne()-i})}if($().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let l=0;l<o.length;l++){const u=o[l];u.data().then(d=>{yf(d,u.dtype,e)})}return{kernelName:e,outputs:o,inputs:t,timeMs:a.then(l=>l.kernelMs),extraInfo:a.then(l=>l.getExtraProfileInfo!=null?l.getExtraProfileInfo():"")}}logKernelProfile(e){const{kernelName:t,outputs:s,timeMs:o,inputs:r,extraInfo:a}=e;s.forEach(i=>{Promise.all([i.data(),o,a]).then(c=>{this.logger.logKernelProfile(t,i,c[0],c[1],r,c[2])})})}}function yf(n,e,t){if(e!=="float32")return!1;for(let s=0;s<n.length;s++){const o=n[s];if(isNaN(o)||!isFinite(o))return console.warn(`Found ${o} in the result of '${t}'`),!0}return!1}class $f{logKernelProfile(e,t,s,o,r,a){const i=typeof o=="number"?bt(`${o}ms`,9):o.error,c=bt(e,25),l=t.rank,u=t.size,d=bt(t.shape.toString(),14);let h="";for(const f in r){const p=r[f];if(p!=null){const x=p.shape||t.shape,g=x.length;h+=`${f}: ${g}D ${g>0?x:""} `}}console.log(`%c${c}	%c${i}	%c${l}D ${d}	%c${u}	%c${h}	%c${a}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vf(n,e,t){const s={},o={};for(let c=0;c<e.length;c++)s[e[c].id]=!0;for(let c=0;c<n.length;c++){const l=n[c],u=l.inputs;for(const d in u){const h=u[d];let f=!1;for(let p=0;p<e.length;p++)if(s[h.id]){l.outputs.forEach(x=>s[x.id]=!0),f=!0,o[l.id]=!0;break}if(f)break}}const r={};r[t.id]=!0;const a={};for(let c=n.length-1;c>=0;c--){const l=n[c],u=l.inputs;for(let d=0;d<l.outputs.length;d++)if(r[l.outputs[d].id]){for(const h in u)r[u[h].id]=!0,a[l.id]=!0;break}}const i=[];for(let c=0;c<n.length;c++){const l=n[c];if(o[l.id]&&a[l.id]){const u={};for(const h in l.inputs){const f=l.inputs[h];s[f.id]&&(u[h]=f)}const d=Object.assign({},l);d.inputs=u,d.outputs=l.outputs,i.push(d)}}return i}function Sf(n,e,t,s){for(let o=e.length-1;o>=0;o--){const r=e[o],a=[];if(r.outputs.forEach(c=>{const l=n[c.id];l!=null?a.push(l):a.push(null)}),r.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${r.kernelName}.`);const i=r.gradient(a);for(const c in r.inputs){if(!(c in i))throw new Error(`Cannot backprop through input ${c}. Available gradients found: ${Object.keys(i)}.`);const l=t(()=>i[c]());if(l.dtype!=="float32")throw new Error(`Error in gradient for op ${r.kernelName}. The gradient of input ${c} must have 'float32' dtype, but has '${l.dtype}'`);const u=r.inputs[c];if(!oe(l.shape,u.shape))throw new Error(`Error in gradient for op ${r.kernelName}. The gradient of input '${c}' has shape '${l.shape}', which does not match the shape of the input '${u.shape}'`);if(n[u.id]==null)n[u.id]=l;else{const d=n[u.id];n[u.id]=s(d,l),d.dispose()}}}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tr=20,dn=3,Ps=7;function If(n,e,t,s){const o=J(e),r=Tf(n,e,t,o),a=e.length,i=Qn(n,e,t,o,r),c=["Tensor"];return s&&(c.push(`  dtype: ${t}`),c.push(`  rank: ${a}`),c.push(`  shape: [${e}]`),c.push("  values:")),c.push(i.map(l=>"    "+l).join(`
`)),c.join(`
`)}function Tf(n,e,t,s){const o=A(e),r=s[s.length-1],a=new Array(r).fill(0),i=e.length,c=t==="complex64"?fn(n):n;if(i>1)for(let l=0;l<o/r;l++){const u=l*r;for(let d=0;d<r;d++)a[d]=Math.max(a[d],hn(c[u+d],0,t).length)}return a}function hn(n,e,t){let s;return Array.isArray(n)?s=`${parseFloat(n[0].toFixed(Ps))} + ${parseFloat(n[1].toFixed(Ps))}j`:Fn(n)?s=`'${n}'`:t==="bool"?s=ti(n):s=parseFloat(n.toFixed(Ps)).toString(),bt(s,e)}function ti(n){return n===0?"false":"true"}function Qn(n,e,t,s,o,r=!0){const a=t==="complex64"?2:1,i=e[0],c=e.length;if(c===0){if(t==="complex64"){const x=fn(n);return[hn(x[0],0,t)]}return t==="bool"?[ti(n[0])]:[n[0].toString()]}if(c===1){if(i>Tr){const g=dn*a;let m=Array.from(n.slice(0,g)),b=Array.from(n.slice((i-dn)*a,i*a));return t==="complex64"&&(m=fn(m),b=fn(b)),["["+m.map((C,y)=>hn(C,o[y],t)).join(", ")+", ..., "+b.map((C,y)=>hn(C,o[i-dn+y],t)).join(", ")+"]"]}return["["+(t==="complex64"?fn(n):Array.from(n)).map((g,m)=>hn(g,o[m],t)).join(", ")+"]"]}const l=e.slice(1),u=s.slice(1),d=s[0]*a,h=[];if(i>Tr){for(let x=0;x<dn;x++){const g=x*d,m=g+d;h.push(...Qn(n.slice(g,m),l,t,u,o,!1))}h.push("...");for(let x=i-dn;x<i;x++){const g=x*d,m=g+d;h.push(...Qn(n.slice(g,m),l,t,u,o,x===i-1))}}else for(let x=0;x<i;x++){const g=x*d,m=g+d;h.push(...Qn(n.slice(g,m),l,t,u,o,x===i-1))}const f=c===2?",":"";h[0]="["+(i>0?h[0]+f:"");for(let x=1;x<h.length-1;x++)h[x]=" "+h[x]+f;let p=`,
`;for(let x=2;x<c;x++)p+=`
`;return h[h.length-1]=" "+h[h.length-1]+"]"+(r?"":p),h}function fn(n){const e=[];for(let t=0;t<n.length;t+=2)e.push([n[t],n[t+1]]);return e}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class is{constructor(e,t,s){if(this.dtype=t,this.shape=e.slice(),this.size=A(e),s!=null){const o=s.length;R(o===this.size,()=>`Length of values '${o}' does not match the size inferred by the shape '${this.size}'.`)}if(t==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=s||te(t,this.size),this.strides=J(e)}set(e,...t){t.length===0&&(t=[0]),R(t.length===this.rank,()=>`The number of provided coordinates (${t.length}) must match the rank (${this.rank})`);const s=this.locToIndex(t);this.values[s]=e}get(...e){e.length===0&&(e=[0]);let t=0;for(const o of e){if(o<0||o>=this.shape[t]){const r=`Requested out of range element at ${e}.   Buffer shape=${this.shape}`;throw new Error(r)}t++}let s=e[e.length-1];for(let o=0;o<e.length-1;++o)s+=this.strides[o]*e[o];return this.values[s]}locToIndex(e){if(this.rank===0)return 0;if(this.rank===1)return e[0];let t=e[e.length-1];for(let s=0;s<e.length-1;++s)t+=this.strides[s]*e[s];return t}indexToLoc(e){if(this.rank===0)return[];if(this.rank===1)return[e];const t=new Array(this.shape.length);for(let s=0;s<t.length-1;++s)t[s]=Math.floor(e/this.strides[s]),e-=t[s]*this.strides[s];return t[t.length-1]=e,t}get rank(){return this.shape.length}toTensor(){return De().makeTensor(this.values,this.shape,this.dtype)}}let De=null,Mt=null;function Rf(n){De=n}function Ef(n){Mt=n}class xe{constructor(e,t,s,o){this.kept=!1,this.isDisposedInternal=!1,this.shape=e.slice(),this.dtype=t||"float32",this.size=A(e),this.strides=J(e),this.dataId=s,this.id=o,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const e=await this.data();return Mt.buffer(this.shape,this.dtype,e)}bufferSync(){return Mt.buffer(this.shape,this.dtype,this.dataSync())}async array(){const e=await this.data();return Ut(this.shape,e,this.dtype==="complex64")}arraySync(){return Ut(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const e=De().read(this.dataId);if(this.dtype==="string"){const t=await e;try{return t.map(s=>Ct(s))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return e}dataToGPU(e){return this.throwIfDisposed(),De().readToGPU(this.dataId,e)}dataSync(){this.throwIfDisposed();const e=De().readSync(this.dataId);if(this.dtype==="string")try{return e.map(t=>Ct(t))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return e}async bytes(){this.throwIfDisposed();const e=await De().read(this.dataId);return this.dtype==="string"?e:new Uint8Array(e.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),De().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(e=!1){return Mt.print(this,e)}clone(){return this.throwIfDisposed(),Mt.clone(this)}toString(e=!1){const t=this.dataSync();return If(t,this.shape,this.dtype,e)}cast(e){return this.throwIfDisposed(),Mt.cast(this,e)}variable(e=!0,t,s){return this.throwIfDisposed(),De().makeVariable(this,e,t,s)}}Object.defineProperty(xe,Symbol.hasInstance,{value:n=>!!n&&n.data!=null&&n.dataSync!=null&&n.throwIfDisposed!=null});function ni(){return mo("Tensor",()=>xe)}ni();class cs extends xe{constructor(e,t,s,o){super(e.shape,e.dtype,e.dataId,o),this.trainable=t,this.name=s}assign(e){if(e.dtype!==this.dtype)throw new Error(`dtype of the new value (${e.dtype}) and previous value (${this.dtype}) must match`);if(!oe(e.shape,this.shape))throw new Error(`shape of the new value (${e.shape}) and previous value (${this.shape}) must match`);De().disposeTensor(this),this.dataId=e.dataId,De().incRef(this,null)}dispose(){De().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(cs,Symbol.hasInstance,{value:n=>n instanceof xe&&n.assign!=null&&n.assign instanceof Function});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Rr;(function(n){n.R0="R0",n.R1="R1",n.R2="R2",n.R3="R3",n.R4="R4",n.R5="R5",n.R6="R6"})(Rr||(Rr={}));var zs;(function(n){n.float32="float32",n.int32="int32",n.bool="int32",n.complex64="complex64"})(zs||(zs={}));var Hs;(function(n){n.float32="float32",n.int32="int32",n.bool="bool",n.complex64="complex64"})(Hs||(Hs={}));var Xs;(function(n){n.float32="float32",n.int32="float32",n.bool="float32",n.complex64="complex64"})(Xs||(Xs={}));var qs;(function(n){n.float32="complex64",n.int32="complex64",n.bool="complex64",n.complex64="complex64"})(qs||(qs={}));const kf={float32:Xs,int32:zs,bool:Hs,complex64:qs};function Ge(n,e){if(n==="string"||e==="string"){if(n==="string"&&e==="string")return"string";throw new Error(`Can not upcast ${n} with ${e}`)}return kf[n][e]}function Po(n){return Ge(n,"int32")}function si(n){return n!=null&&typeof n=="object"&&"texture"in n&&n.texture instanceof WebGLTexture}function oi(n){return typeof GPUBuffer<"u"&&n!=null&&typeof n=="object"&&"buffer"in n&&n.buffer instanceof GPUBuffer}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ut(n,e){if(n.dtype===e.dtype)return[n,e];const t=Ge(n.dtype,e.dtype);return[n.cast(t),e.cast(t)]}function Nf(n,e){R(n.dtype===e.dtype,()=>`The dtypes of the first(${n.dtype}) and second(${e.dtype}) input must match`)}function Af(n,e){return e.some(t=>t.id===n.id)}function _o(n){const e=[];return ri(n,e,new Set),e}function ri(n,e,t){if(n==null)return;if(n instanceof xe){e.push(n);return}if(!Ff(n))return;const s=n;for(const o in s){const r=s[o];t.has(r)||(t.add(r),ri(r,e,t))}}function Ff(n){return Array.isArray(n)||typeof n=="object"}const WT=Object.freeze(Object.defineProperty({__proto__:null,assertTypesMatch:Nf,getTensorsInContainer:_o,isTensorInList:Af,makeTypesMatch:ut},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _s(n){return n.kernelName!=null}class Er{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(e=>e.name)))}}}dispose(){for(const e in this.registeredVariables)this.registeredVariables[e].dispose()}}class qt{constructor(e){this.ENV=e,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new Er}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const e=this.getSortedBackends();for(let t=0;t<e.length;t++){const s=e[t];if(await this.initializeBackend(s).success){await this.setBackend(s);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:e,asyncInit:t}=this.initializeBackendsAndReturnBest();if(t)throw new Error(`The highest priority backend '${e}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(e)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(e){if(!(e in this.registry))if(e in this.registryFactory){const{asyncInit:t}=this.initializeBackend(e);if(t)return null}else return null;return this.registry[e]}findBackendFactory(e){return e in this.registryFactory?this.registryFactory[e].factory:null}registerBackend(e,t,s=1){return e in this.registryFactory?(Ae(`${e} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[e]={factory:t,priority:s},!0)}async setBackend(e){if(this.registryFactory[e]==null)throw new Error(`Backend name '${e}' not found in registry`);if(this.backendName=e,this.registry[e]==null){this.backendInstance=null;const{success:t,asyncInit:s}=this.initializeBackend(e);if(!(s?await t:t))return!1}return this.backendInstance=this.registry[e],this.setupRegisteredKernels(),this.profiler=new Cf(this.backendInstance),!0}setupRegisteredKernels(){Ws(this.backendName).forEach(t=>{t.setupFunc!=null&&t.setupFunc(this.backendInstance)})}disposeRegisteredKernels(e){Ws(e).forEach(s=>{s.disposeFunc!=null&&s.disposeFunc(this.registry[e])})}initializeBackend(e){const t=this.registryFactory[e];if(t==null)throw new Error(`Cannot initialize backend ${e}, no registration found.`);try{const s=t.factory();if(s&&!(s instanceof ga)&&typeof s.then=="function"){const o=++this.pendingBackendInitId,r=s.then(a=>o<this.pendingBackendInitId?!1:(this.registry[e]=a,this.pendingBackendInit=null,!0)).catch(a=>(o<this.pendingBackendInitId||(this.pendingBackendInit=null,Ae(`Initialization of backend ${e} failed`),Ae(a.stack||a.message)),!1));return this.pendingBackendInit=r,{success:r,asyncInit:!0}}else return this.registry[e]=s,{success:!0,asyncInit:!1}}catch(s){return Ae(`Initialization of backend ${e} failed`),Ae(s.stack||s.message),{success:!1,asyncInit:!1}}}removeBackend(e){if(!(e in this.registryFactory))throw new Error(`${e} backend not found in registry`);this.backendName===e&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,e in this.registry&&(this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e]),delete this.registryFactory[e],this.backendName===e&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((e,t)=>this.registryFactory[t].priority-this.registryFactory[e].priority)}initializeBackendsAndReturnBest(){const e=this.getSortedBackends();for(let t=0;t<e.length;t++){const s=e[t],{success:o,asyncInit:r}=this.initializeBackend(s);if(r||o)return{name:s,asyncInit:r}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(e,t){const s=this.state.tensorInfo.get(t),o=s.backend,r=this.readSync(t),a=o.refCount(t);o.disposeData(t,!0),s.backend=e,e.move(t,r,s.shape,s.dtype,a),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(e,t){let s=null;if(t==null){if(typeof e!="function")throw new Error("Please provide a function to tidy()");t=e}else{if(typeof e!="string"&&!(e instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof t!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");s=e}let o;return this.scopedRun(()=>this.startScope(s),()=>this.endScope(o),()=>(o=t(),o instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),o))}scopedRun(e,t,s){e();try{const o=s();return t(),o}catch(o){throw t(),o}}nextTensorId(){return qt.nextTensorId++}nextVariableId(){return qt.nextVariableId++}clone(e){const t=O.runKernel(bs,{x:e}),s={x:e},o=a=>({x:()=>{const i="float32",c={x:a},l={dtype:i};return O.runKernel(gs,c,l)}}),r=[];return this.addTapeNode(this.state.activeScope.name,s,[t],o,r,{}),t}runKernel(e,t,s){if(this.backendName==null&&this.backend,!($r(e,this.backendName)!=null))throw new Error(`Kernel '${e}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:e,inputs:t,attrs:s})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(e,t,s){const o=this.backend.numDataIds();let r=0;s.forEach(c=>{r+=c.dtype==="complex64"?3:1});const a=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],i=o-t-r-a;if(i>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${i} data ids) after running '${e}'`)}runKernelFunc(e){let t,s=[];const o=this.isTapeOn(),r=this.state.numBytes,a=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let i;this.backendName==null&&this.backend;let c;const l=_s(e)?e.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(_s(e)){const{kernelName:p,inputs:x,attrs:g}=e;this.backendName==null&&this.backend;const m=$r(p,this.backendName);R(m!=null,()=>`Cannot find registered kernel '${p}' for backend '${this.backendName}'`),i=()=>{const b=this.backend.numDataIds();c=m.kernelFunc({inputs:x,attrs:g,backend:this.backend});const C=Array.isArray(c)?c:[c];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(p,b,C);const y=C.map(S=>S.rank!=null?S:this.makeTensorFromTensorInfo(S));if(o){const S=this.getTensorsForGradient(p,x,y);s=this.saveTensorsForBackwardMode(S)}return y}}else{const{forwardFunc:p}=e,x=g=>{o&&(s=g.map(m=>this.keep(this.clone(m))))};i=()=>{const g=this.backend.numDataIds();c=this.tidy(()=>p(this.backend,x));const m=Array.isArray(c)?c:[c];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(l,g,m),m}}const{inputs:u,attrs:d}=e,h=_s(e)?null:e.backwardsFunc;let f;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?t=i():(f=this.profiler.profileKernel(l,u,()=>i()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(f),t=f.outputs)}),o&&this.addTapeNode(l,u,t,h,s,d),this.state.profiling&&this.state.activeProfile.kernels.push({name:l,bytesAdded:this.state.numBytes-r,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-a,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(u).map(p=>u[p]!=null?u[p].shape:null),outputShapes:t.map(p=>p.shape),kernelTimeMs:f.timeMs,extraInfo:f.extraInfo}),Array.isArray(c)?t:t[0]}saveTensorsForBackwardMode(e){return e.map(s=>this.keep(this.clone(s)))}getTensorsForGradient(e,t,s){const o=vr(e);if(o!=null){const r=o.inputsToSave||[],a=o.outputsToSave||[];let i;o.saveAllInputs?(R(Array.isArray(t),()=>"saveAllInputs is true, expected inputs to be an array."),i=Object.keys(t).map(l=>t[l])):i=r.map(l=>t[l]);const c=s.filter((l,u)=>a[u]);return i.concat(c)}return[]}makeTensor(e,t,s,o){if(e==null)throw new Error("Values passed to engine.makeTensor() are null");s=s||"float32",o=o||this.backend;let r=e;s==="string"&&Fn(e[0])&&(r=e.map(c=>ot(c)));const a=o.write(r,t,s),i=new xe(t,s,a,this.nextTensorId());if(this.trackTensor(i,o),s==="string"){const c=this.state.tensorInfo.get(a),l=$a(r);this.state.numBytes+=l-c.bytes,c.bytes=l}return i}makeTensorFromDataId(e,t,s,o){s=s||"float32";const r={dataId:e,shape:t,dtype:s};return this.makeTensorFromTensorInfo(r,o)}makeTensorFromTensorInfo(e,t){const{dataId:s,shape:o,dtype:r}=e,a=new xe(o,r,s,this.nextTensorId());return this.trackTensor(a,t),a}makeVariable(e,t=!0,s,o){s=s||this.nextVariableId().toString(),o!=null&&o!==e.dtype&&(e=e.cast(o));const r=new cs(e,t,s,this.nextTensorId());if(this.state.registeredVariables[r.name]!=null)throw new Error(`Variable with name ${r.name} was already registered`);return this.state.registeredVariables[r.name]=r,this.incRef(r,this.backend),r}trackTensor(e,t){this.state.numTensors++,e.dtype==="string"&&this.state.numStringTensors++;let s=0;e.dtype!=="complex64"&&e.dtype!=="string"&&(s=e.size*yn(e.dtype)),this.state.numBytes+=s,this.state.tensorInfo.has(e.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(e.dataId,{backend:t||this.backend,dtype:e.dtype,shape:e.shape,bytes:s})),e instanceof cs||this.track(e)}incRef(e,t){this.trackTensor(e,t),this.backend.incRef(e.dataId)}removeDataId(e,t){this.state.tensorInfo.has(e)&&this.state.tensorInfo.get(e).backend===t&&(this.state.tensorInfo.delete(e),this.state.numDataBuffers--)}disposeTensor(e){if(!this.state.tensorInfo.has(e.dataId))return;const t=this.state.tensorInfo.get(e.dataId);if(this.state.numTensors--,e.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=t.bytes),e.dtype!=="complex64"&&e.dtype!=="string"){const s=e.size*yn(e.dtype);this.state.numBytes-=s}t.backend.disposeData(e.dataId)&&this.removeDataId(e.dataId,t.backend)}disposeVariables(){for(const e in this.state.registeredVariables){const t=this.state.registeredVariables[e];this.disposeVariable(t)}}disposeVariable(e){this.disposeTensor(e),this.state.registeredVariables[e.name]!=null&&delete this.state.registeredVariables[e.name]}memory(){const e=this.backend.memory();return e.numTensors=this.state.numTensors,e.numDataBuffers=this.state.numDataBuffers,e.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(e.unreliable=!0,e.reasons==null&&(e.reasons=[]),e.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),e}async profile(e){this.state.profiling=!0;const t=this.state.numBytes,s=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await e(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(o=>o.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-t,this.state.activeProfile.newTensors=this.state.numTensors-s;for(const o of this.state.activeProfile.kernels)o.kernelTimeMs=await o.kernelTimeMs,o.extraInfo=await o.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(e,t,s,o,r,a){const i={id:this.state.nextTapeNodeId++,kernelName:e,inputs:t,outputs:s,saved:r},c=vr(e);c!=null&&(o=c.gradFunc),o!=null&&(i.gradient=l=>(l=l.map((u,d)=>{if(u==null){const h=s[d],f=We(h.size,h.dtype);return this.makeTensor(f,h.shape,h.dtype)}return u}),o(l.length>1?l:l[0],r,a))),this.state.activeTape.push(i)}keep(e){return e.kept=!0,e}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(e){const t={track:[],name:"unnamed scope",id:this.state.nextScopeId++};e&&(t.name=e),this.state.scopeStack.push(t),this.state.activeScope=t}endScope(e){const t=_o(e),s=new Set(t.map(r=>r.id));for(let r=0;r<this.state.activeScope.track.length;r++){const a=this.state.activeScope.track[r];!a.kept&&!s.has(a.id)&&a.dispose()}const o=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],t.forEach(r=>{!r.kept&&r.scopeId===o.id&&this.track(r)})}gradients(e,t,s,o=!1){if(R(t.length>0,()=>"gradients() received an empty list of xs."),s!=null&&s.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${s.dtype}'`);const r=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",e));R(r instanceof xe,()=>"The result y returned by f() must be a tensor.");const a=vf(this.state.activeTape,t,r);if(!o&&a.length===0&&t.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const i={};i[r.id]=s??Df(r.shape),Sf(i,a,l=>this.tidy(l),Of);const c=t.map(l=>i[l.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(l=>{for(const u of l.saved)u.dispose()}),this.state.activeTape=null),{value:r,grads:c}})}customGrad(e){return R(rt(e),()=>"The f passed in customGrad(f) must be a function."),(...t)=>{R(t.every(i=>i instanceof xe),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let s;const o={};t.forEach((i,c)=>{o[c]=i});const r=(i,c)=>(s=e(...t,c),R(s.value instanceof xe,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),R(rt(s.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),s.value),a=(i,c)=>{const l=s.gradFunc(i,c),u=Array.isArray(l)?l:[l];R(u.length===t.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),R(u.every(h=>h instanceof xe),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const d={};return u.forEach((h,f)=>{d[f]=()=>h}),d};return this.runKernelFunc({forwardFunc:r,backwardsFunc:a,inputs:o})}}readSync(e){return this.state.tensorInfo.get(e).backend.readSync(e)}read(e){return this.state.tensorInfo.get(e).backend.read(e)}readToGPU(e,t){return this.state.tensorInfo.get(e).backend.readToGPU(e,t)}async time(e){const t=Ne(),s=await this.backend.time(e);return s.wallMs=Ne()-t,s}track(e){return this.state.activeScope!=null&&(e.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(e)),e}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new Er;for(const e in this.registry)this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}qt.nextTensorId=0;qt.nextVariableId=0;function Df(n){const e=Ta(A(n),"float32");return O.makeTensor(e,n,"float32")}function ai(){const n=Ea();if(n._tfengine==null){const e=new Wu(n);n._tfengine=new qt(e)}return Xu(n._tfengine.ENV),Rf(()=>n._tfengine),n._tfengine}const O=ai();function Of(n,e){const t={a:n,b:e};return O.runKernel(On,t)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pf(){return typeof navigator<"u"&&navigator!=null}let js;function _f(n){js=n}function Lo(n){if(js!==void 0)return js;if(n||Pf()){if(n||(n=navigator),n.product==="ReactNative")return!0;const e=n.userAgent||n.vendor||(typeof window<"u"?window.opera:"");if(!e){const t=n;return t.userAgentData&&t.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(e)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(e.substr(0,4))}return!1}function Bo(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}const GT=Object.freeze(Object.defineProperty({__proto__:null,isBrowser:Bo,isMobile:Lo,mockIsMobile:_f},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const be=$();be.registerFlag("DEBUG",()=>!1,n=>{n&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});be.registerFlag("IS_BROWSER",()=>Bo());be.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");be.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));be.registerFlag("IS_SAFARI",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor));be.registerFlag("PROD",()=>!1);be.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>be.getBool("DEBUG"));be.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);be.registerFlag("IS_TEST",()=>!1);be.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>be.getBool("DEBUG"));be.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);be.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);be.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ii(n,e){let t=n;if(Te(n))return e==="string"?[]:[n.length];if(si(n)){const o=n.channels||"RGBA";return[n.height,n.width*o.length]}else if(oi(n))return[n.buffer.size/(e==null?4:yn(e))];if(!Array.isArray(n))return[];const s=[];for(;Array.isArray(t)||Te(t)&&e!=="string";)s.push(t.length),t=t[0];return Array.isArray(n)&&$().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&ci(n,s,[]),s}function ci(n,e,t){if(t=t||[],!Array.isArray(n)&&!Te(n)){R(e.length===0,()=>`Element arr[${t.join("][")}] is a primitive, but should be an array/TypedArray of ${e[0]} elements`);return}R(e.length>0,()=>`Element arr[${t.join("][")}] should be a primitive, but is an array of ${n.length} elements`),R(n.length===e[0],()=>`Element arr[${t.join("][")}] should have ${e[0]} elements, but has ${n.length} elements`);const s=e.slice(1);for(let o=0;o<n.length;++o)ci(n[o],s,t.concat(o))}function kr(n,e,t,s){if(n!=="string_or_numeric"){if(n==null)throw new Error("Expected dtype cannot be null.");if(n!=="numeric"&&n!==e||n==="numeric"&&e==="string")throw new Error(`Argument '${t}' passed to '${s}' must be ${n} tensor, but got ${e} tensor`)}}function M(n,e,t,s="numeric"){if(n instanceof ni())return kr(s,n.dtype,e,t),n;let o=Yt(n);if(o!=="string"&&["bool","int32","float32"].indexOf(s)>=0&&(o=s),kr(s,o,e,t),n==null||!Te(n)&&!Array.isArray(n)&&typeof n!="number"&&typeof n!="boolean"&&typeof n!="string"){const c=n==null?"null":n.constructor.name;throw new Error(`Argument '${e}' passed to '${t}' must be a Tensor or TensorLike, but got '${c}'`)}const r=ii(n,o);!Te(n)&&!Array.isArray(n)&&(n=[n]);const i=o!=="string"?_n(n,o):at(n,[],!0);return O.makeTensor(i,r,o)}function Lf(n,e,t,s="numeric"){if(!Array.isArray(n))throw new Error(`Argument ${e} passed to ${t} must be a \`Tensor[]\` or \`TensorLike[]\``);return n.map((r,a)=>M(r,`${e}[${a}]`,t,s))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bf="__op";function Q(n){const e=Object.keys(n);if(e.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${e.length} keys.`);let t=e[0];const s=n[t];t.endsWith("_")&&(t=t.substring(0,t.length-1)),t=t+Bf;const o=(...r)=>{O.startScope(t);try{const a=s(...r);return ms(a)&&console.error("Cannot return a Promise inside of tidy."),O.endScope(a),a}catch(a){throw O.endScope(null),a}};return Object.defineProperty(o,"name",{value:t,configurable:!0}),o}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mf(n,e){const t=M(n,"real","complex"),s=M(e,"imag","complex");Kt(t.shape,s.shape,`real and imag shapes, ${t.shape} and ${s.shape}, must match in call to tf.complex().`);const o={real:t,imag:s};return O.runKernel(wo,o)}const li=Q({complex_:Mf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ui(n,e,t,s){if(s==null)s=Yt(n);else if(s==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(oi(n)||si(n)){if(s!=="float32"&&s!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${s}.`);return O.backend.createTensorFromGPUData(n,e||t,s)}if(!Te(n)&&!Array.isArray(n)&&typeof n!="number"&&typeof n!="boolean"&&typeof n!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(e!=null){Qt(e);const o=A(e),r=A(t);R(o===r,()=>`Based on the provided shape, [${e}], the tensor should have ${o} values but has ${r}`);for(let a=0;a<t.length;++a){const i=t[a],c=a===t.length-1?i!==A(e.slice(a)):!0;R(t[a]===e[a]||!c,()=>`Error creating a new Tensor. Inferred shape (${t}) does not match the provided shape (${e}). `)}}return!Te(n)&&!Array.isArray(n)&&(n=[n]),e=e||t,n=s!=="string"?_n(n,s):at(n,[],!0),O.makeTensor(n,e,s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ls(n,e,t){const s=ii(n,t);return ui(n,e,s,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yt={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};class ze{static join(e){return new ze(e).slice()}constructor(e){if(this.shards=[],this.previousShardIndex=0,e==null||(e instanceof Array||(e=[e]),e=e.map(s=>Te(s)?s.buffer:s),e.length===0))return;this.bufferUniformSize=e[0].byteLength;let t=0;for(let s=0;s<e.length;s++){const o=e[s];s!==e.length-1&&o.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const r=t+o.byteLength;this.shards.push({buffer:o,start:t,end:r}),t=r}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(e=0,t=this.byteLength){if(this.shards.length===0)return new ArrayBuffer(0);if(e=isNaN(Number(e))?0:e,t=isNaN(Number(t))?0:t,e=Math.max(0,e),t=Math.min(this.byteLength,t),t<=e)return new ArrayBuffer(0);const s=this.findShardForByte(e);if(s===-1)throw new Error(`Could not find start shard for byte ${e}`);const o=t-e,r=new ArrayBuffer(o),a=new Uint8Array(r);let i=0;for(let c=s;c<this.shards.length;c++){const l=this.shards[c],d=e+i-l.start,h=i,p=Math.min(t,l.end)-l.start,x=new Uint8Array(l.buffer,d,p-d);if(a.set(x,h),i+=x.length,t<l.end)break}return r}findShardForByte(e){if(this.shards.length===0||e<0||e>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(e/this.bufferUniformSize),this.previousShardIndex;function t(o){return e<o.start?-1:e>=o.end?1:0}if(t(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;const s=Vf(this.shards,t);return s===-1?-1:(this.previousShardIndex=s,this.previousShardIndex)}}function Vf(n,e){let t=0,s=n.length;for(;t<=s;){const o=Math.floor((s-t)/2)+t,r=e(n[o]);if(r===0)return o;r<0?s=o:t=o+1}return-1}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zT(){$().set("PROD",!0)}function HT(){$().set("DEBUG",!0)}function XT(){$().set("DEPRECATION_WARNINGS_ENABLED",!1),console.warn("TensorFlow.js deprecation warnings have been disabled.")}function qT(n){$().getBool("DEPRECATION_WARNINGS_ENABLED")&&console.warn(n+" You can disable deprecation warnings with tf.disableDeprecationWarnings().")}function jT(){O.disposeVariables()}function Ze(){return O}function KT(){return O.memory()}function YT(n){return O.profile(n)}function ee(n,e){return O.tidy(n,e)}function Ie(n){_o(n).forEach(t=>t.dispose())}function Uf(n){return O.keep(n)}function QT(n){return O.time(n)}function ZT(n){return O.setBackend(n)}function JT(){return O.ready()}function Wf(){return O.backendName}function eR(n){O.removeBackend(n)}function tR(n){return O.findBackend(n)}function nR(n){return O.findBackendFactory(n)}function Gf(n,e,t=1){return O.registerBackend(n,e,t)}function zf(){return O.backend}function sR(n,e){$().setPlatform(n,e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const it=4;async function oR(n,e){const t=[],s=[],o=Array.isArray(n)?n.map(a=>a.name):Object.keys(n);for(let a=0;a<o.length;++a){const i=o[a],c=Array.isArray(n)?n[a].tensor:n[i];if(c.dtype!=="float32"&&c.dtype!=="int32"&&c.dtype!=="bool"&&c.dtype!=="string"&&c.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${i}': ${c.dtype}`);const l={name:i,shape:c.shape,dtype:c.dtype};if(c.dtype==="string"){const u=new Promise(async d=>{const h=await c.bytes(),f=h.reduce((g,m)=>g+m.length,0)+it*h.length,p=new Uint8Array(f);let x=0;for(let g=0;g<h.length;g++){const m=h[g],b=new Uint8Array(new Uint32Array([m.length]).buffer);p.set(b,x),x+=it,p.set(m,x),x+=m.length}d(p)});s.push(u)}else s.push(c.data());e!=null&&(l.group=e),t.push(l)}const r=await Promise.all(s);return{data:jf(r),specs:t}}function Hf(n,e){const t=new ze(n),s={};let o=0;for(const r of e){const a=Xf(r,(i,c)=>t.slice(o+i,o+c));s[r.name]=di(r,t.slice(o,o+a)),o+=a}return s}function Xf(n,e){const t=A(n.shape);let s;if("quantization"in n){const o=n.quantization;s=yt[o.dtype]}else if(n.dtype==="string"){let o=0;for(let r=0;r<t;r++)o+=it+new Uint32Array(e(o,o+it))[0];return o}else s=yt[n.dtype];return t*s}async function qf(n,e){const t=A(n.shape);let s;if("quantization"in n){const o=n.quantization;s=yt[o.dtype]}else if(n.dtype==="string"){let o=0;for(let r=0;r<t;r++)o+=it+new Uint32Array(await e(o,o+it))[0];return o}else s=yt[n.dtype];return t*s}function di(n,e){const t=n.name,s=n.dtype,o=n.shape,r=A(o);let a,i=0;if("quantization"in n){const c=n.quantization;if(c.dtype==="uint8"||c.dtype==="uint16"){if(!("min"in c&&"scale"in c))throw new Error(`Weight ${n.name} with quantization ${c.dtype} doesn't have corresponding metadata min and scale.`)}else if(c.dtype==="float16"){if(s!=="float32")throw new Error(`Weight ${n.name} is quantized with ${c.dtype} which only supports weights of type float32 not ${s}.`)}else throw new Error(`Weight ${n.name} has unknown quantization dtype ${c.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);const l=yt[c.dtype],u=c.dtype==="uint8"?new Uint8Array(e):new Uint16Array(e);if(s==="float32")if(c.dtype==="uint8"||c.dtype==="uint16"){a=new Float32Array(u.length);for(let d=0;d<u.length;d++){const h=u[d];a[d]=h*c.scale+c.min}}else if(c.dtype==="float16")a=tp()(u);else throw new Error(`Unsupported quantization type ${c.dtype} for weight type float32.`);else if(s==="int32"){if(c.dtype!=="uint8"&&c.dtype!=="uint16")throw new Error(`Unsupported quantization type ${c.dtype} for weight type int32.`);a=new Int32Array(u.length);for(let d=0;d<u.length;d++){const h=u[d];a[d]=Math.round(h*c.scale+c.min)}}else throw new Error(`Unsupported dtype in weight '${t}': ${s}`);i+=r*l}else if(s==="string"){const c=A(n.shape);a=[];for(let l=0;l<c;l++){const u=new Uint32Array(e.slice(i,i+it))[0];i+=it;const d=new Uint8Array(e.slice(i,i+u));a.push(d),i+=u}}else{const c=yt[s];if(s==="float32")a=new Float32Array(e);else if(s==="int32")a=new Int32Array(e);else if(s==="bool")a=new Uint8Array(e);else if(s==="complex64"){a=new Float32Array(e);const l=new Float32Array(a.length/2),u=new Float32Array(a.length/2);for(let p=0;p<l.length;p++)l[p]=a[p*2],u[p]=a[p*2+1];const d=Ls(l,o,"float32"),h=Ls(u,o,"float32"),f=li(d,h);return d.dispose(),h.dispose(),f}else throw new Error(`Unsupported dtype in weight '${t}': ${s}`);i+=r*c}return Ls(a,o,s)}async function Nr(n,e,t){let s=new Uint8Array(e);for(;s.byteLength<t;){const{done:o,value:r}=await n.read();if(o&&r==null){const i=t-s.byteLength;throw new Error(`Reader is done but ${i} bytes are still expected`)}const a=new Uint8Array(s.length+r.byteLength);a.set(s,0),a.set(new Uint8Array(r),s.length),s=a}return s.buffer}async function rR(n,e){const t={},s=n.getReader();let o=new ArrayBuffer(0);for(const r of e){const a=await qf(r,async(l,u)=>(o=await Nr(s,o,u),o.slice(l,u)));o=await Nr(s,o,a);const i=o.slice(0,a);o=o.slice(a);const c=di(r,i);if(t[r.name]=c,Wf()==="webgpu"){const l=zf();"uploadToGPU"in l&&A(c.shape)>=$().get("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD")&&l.uploadToGPU(c.dataId)}}return t}function jf(n){if(n===null)throw new Error(`Invalid input value: ${JSON.stringify(n)}`);let e=0;const t=[];n.forEach(r=>{if(e+=r.byteLength,t.push(r.byteLength===r.buffer.byteLength?r:new r.constructor(r)),!(r instanceof Float32Array||r instanceof Int32Array||r instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${r.constructor.name}`)});const s=new Uint8Array(e);let o=0;return t.forEach(r=>{s.set(new Uint8Array(r.buffer),o),o+=r.byteLength}),s.buffer}const Mo=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function Ar(n){return Mo?Buffer.byteLength(n,"utf8"):new Blob([n]).size}function Kf(n){if(Mo)return Buffer.from(n).toString("base64");const e=new Uint8Array(n);let t="";for(let s=0,o=e.length;s<o;s++)t+=String.fromCharCode(e[s]);return btoa(t)}function Yf(n){if(Mo){const s=Buffer.from(n,"base64");return s.buffer.slice(s.byteOffset,s.byteOffset+s.byteLength)}const e=atob(n),t=new Uint8Array(e.length);for(let s=0;s<e.length;++s)t.set([e.charCodeAt(s)],s);return t.buffer}function aR(n){return ze.join(n)}function Fr(n){for(n=n.trim();n.endsWith("/");)n=n.slice(0,n.length-1);const t=n.split("/");return t[t.length-1]}function hi(n,e){const t={modelTopology:n.modelTopology,format:n.format,generatedBy:n.generatedBy,convertedBy:n.convertedBy,weightsManifest:e};return n.signature!=null&&(t.signature=n.signature),n.userDefinedMetadata!=null&&(t.userDefinedMetadata=n.userDefinedMetadata),n.modelInitializer!=null&&(t.modelInitializer=n.modelInitializer),n.initializerSignature!=null&&(t.initializerSignature=n.initializerSignature),n.trainingConfig!=null&&(t.trainingConfig=n.trainingConfig),t}function Qf(n,e,t){const s={modelTopology:n.modelTopology,format:n.format,generatedBy:n.generatedBy,convertedBy:n.convertedBy};if(n.trainingConfig!=null&&(s.trainingConfig=n.trainingConfig),n.weightsManifest!=null){if(!e)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!t)throw new Error("modelJSON has weightsManifest but weightData is null");s.weightSpecs=e,s.weightData=t}return n.signature!=null&&(s.signature=n.signature),n.userDefinedMetadata!=null&&(s.userDefinedMetadata=n.userDefinedMetadata),n.modelInitializer!=null&&(s.modelInitializer=n.modelInitializer),n.initializerSignature!=null&&(s.initializerSignature=n.initializerSignature),s}async function fi(n,e){let t,s;return n.weightsManifest!=null&&([t,s]=await e(n.weightsManifest)),Qf(n,t,s)}function Ss(n){if(n.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:n.modelTopology==null?0:Ar(JSON.stringify(n.modelTopology)),weightSpecsBytes:n.weightSpecs==null?0:Ar(JSON.stringify(n.weightSpecs)),weightDataBytes:n.weightData==null?0:new ze(n.weightData).byteLength}}function Dr(n){const e=[];for(const t of n)e.push(...t.weights);return e}function Zf(){const n=t=>{let s=t<<13,o=0;for(;(s&8388608)===0;)o-=8388608,s<<=1;return s&=-8388609,o+=947912704,s|o},e=new Uint32Array(2048);e[0]=0;for(let t=1;t<1024;t++)e[t]=n(t);for(let t=1024;t<2048;t++)e[t]=939524096+(t-1024<<13);return e}function Jf(){const n=new Uint32Array(64);n[0]=0,n[31]=1199570944,n[32]=2147483648,n[63]=3347054592;for(let e=1;e<31;e++)n[e]=e<<23;for(let e=33;e<63;e++)n[e]=2147483648+(e-32<<23);return n}function ep(){const n=new Uint32Array(64);for(let e=0;e<64;e++)n[e]=1024;return n[0]=n[32]=0,n}function tp(){const n=Zf(),e=Jf(),t=ep();return s=>{const o=new ArrayBuffer(4*s.length),r=new Uint32Array(o);for(let a=0;a<s.length;a++){const i=s[a],c=n[t[i>>10]+(i&1023)]+e[i>>10];r[a]=c}return new Float32Array(o)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Y{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return Y.instance==null&&(Y.instance=new Y),Y.instance}static registerSaveRouter(e){Y.getInstance().saveRouters.push(e)}static registerLoadRouter(e){Y.getInstance().loadRouters.push(e)}static getSaveHandlers(e){return Y.getHandlers(e,"save")}static getLoadHandlers(e,t){return Y.getHandlers(e,"load",t)}static getHandlers(e,t,s){const o=[];return(t==="load"?Y.getInstance().loadRouters:Y.getInstance().saveRouters).forEach(a=>{const i=a(e,s);i!==null&&o.push(i)}),o}}const iR=n=>Y.registerSaveRouter(n),cR=n=>Y.registerLoadRouter(n),lR=n=>Y.getSaveHandlers(n),uR=(n,e)=>Y.getLoadHandlers(n,e);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ks="tensorflowjs",Ys=1,xt="models_store",tt="model_info_store";function pi(){if(!$().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const n=typeof window>"u"?self:window,e=n.indexedDB||n.mozIndexedDB||n.webkitIndexedDB||n.msIndexedDB||n.shimIndexedDB;if(e==null)throw new Error("The current browser does not appear to support IndexedDB.");return e}function Qs(n){const e=n.result;e.createObjectStore(xt,{keyPath:"modelPath"}),e.createObjectStore(tt,{keyPath:"modelPath"})}class $t{constructor(e){if(this.indexedDB=pi(),e==null||!e)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=e}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,e)}async load(){return this.databaseAction(this.modelPath)}databaseAction(e,t){return new Promise((s,o)=>{const r=this.indexedDB.open(Ks,Ys);r.onupgradeneeded=()=>Qs(r),r.onsuccess=()=>{const a=r.result;if(t==null){const i=a.transaction(xt,"readonly"),l=i.objectStore(xt).get(this.modelPath);l.onsuccess=()=>{if(l.result==null)return a.close(),o(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));s(l.result.modelArtifacts)},l.onerror=u=>(a.close(),o(l.error)),i.oncomplete=()=>a.close()}else{t.weightData=ze.join(t.weightData);const i=Ss(t),c=a.transaction(tt,"readwrite");let l=c.objectStore(tt),u;try{u=l.put({modelPath:this.modelPath,modelArtifactsInfo:i})}catch(h){return o(h)}let d;u.onsuccess=()=>{d=a.transaction(xt,"readwrite");const h=d.objectStore(xt);let f;try{f=h.put({modelPath:this.modelPath,modelArtifacts:t,modelArtifactsInfo:i})}catch(p){return o(p)}f.onsuccess=()=>s({modelArtifactsInfo:i}),f.onerror=p=>{l=c.objectStore(tt);const x=l.delete(this.modelPath);x.onsuccess=()=>(a.close(),o(f.error)),x.onerror=g=>(a.close(),o(f.error))}},u.onerror=h=>(a.close(),o(u.error)),c.oncomplete=()=>{d==null?a.close():d.oncomplete=()=>a.close()}}},r.onerror=a=>o(r.error)})}}$t.URL_SCHEME="indexeddb://";const mi=n=>$().getBool("IS_BROWSER")&&!Array.isArray(n)&&n.startsWith($t.URL_SCHEME)?np(n.slice($t.URL_SCHEME.length)):null;Y.registerSaveRouter(mi);Y.registerLoadRouter(mi);function np(n){return new $t(n)}function sp(n){return n.startsWith($t.URL_SCHEME)?n.slice($t.URL_SCHEME.length):n}class op{constructor(){this.indexedDB=pi()}async listModels(){return new Promise((e,t)=>{const s=this.indexedDB.open(Ks,Ys);s.onupgradeneeded=()=>Qs(s),s.onsuccess=()=>{const o=s.result,r=o.transaction(tt,"readonly"),i=r.objectStore(tt).getAll();i.onsuccess=()=>{const c={};for(const l of i.result)c[l.modelPath]=l.modelArtifactsInfo;e(c)},i.onerror=c=>(o.close(),t(i.error)),r.oncomplete=()=>o.close()},s.onerror=o=>t(s.error)})}async removeModel(e){return e=sp(e),new Promise((t,s)=>{const o=this.indexedDB.open(Ks,Ys);o.onupgradeneeded=()=>Qs(o),o.onsuccess=()=>{const r=o.result,a=r.transaction(tt,"readwrite"),i=a.objectStore(tt),c=i.get(e);let l;c.onsuccess=()=>{if(c.result==null)return r.close(),s(new Error(`Cannot find model with path '${e}' in IndexedDB.`));{const u=i.delete(e),d=()=>{l=r.transaction(xt,"readwrite");const f=l.objectStore(xt).delete(e);f.onsuccess=()=>t(c.result.modelArtifactsInfo),f.onerror=p=>s(c.error)};u.onsuccess=d,u.onerror=h=>(d(),r.close(),s(c.error))}},c.onerror=u=>(r.close(),s(c.error)),a.oncomplete=()=>{l==null?r.close():l.oncomplete=()=>r.close()}},o.onerror=r=>s(o.error)})}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ke="/",Vt="tensorflowjs_models",gi="info",rp="model_topology",ap="weight_specs",ip="weight_data",cp="model_metadata";function xi(n){return{info:[Vt,n,gi].join(Ke),topology:[Vt,n,rp].join(Ke),weightSpecs:[Vt,n,ap].join(Ke),weightData:[Vt,n,ip].join(Ke),modelMetadata:[Vt,n,cp].join(Ke)}}function bi(n){for(const e of Object.values(n))window.localStorage.removeItem(e)}function lp(n){const e=n.split(Ke);if(e.length<3)throw new Error(`Invalid key format: ${n}`);return e.slice(1,e.length-1).join(Ke)}function up(n){return n.startsWith(vt.URL_SCHEME)?n.slice(vt.URL_SCHEME.length):n}class vt{constructor(e){if(!$().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,e==null||!e)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=e,this.keys=xi(this.modelPath)}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const t=JSON.stringify(e.modelTopology),s=JSON.stringify(e.weightSpecs),o=Ss(e),r=ze.join(e.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(o)),this.LS.setItem(this.keys.topology,t),this.LS.setItem(this.keys.weightSpecs,s),this.LS.setItem(this.keys.weightData,Kf(r));const a={format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,signature:e.signature!=null?e.signature:void 0,userDefinedMetadata:e.userDefinedMetadata!=null?e.userDefinedMetadata:void 0,modelInitializer:e.modelInitializer!=null?e.modelInitializer:void 0,initializerSignature:e.initializerSignature!=null?e.initializerSignature:void 0,trainingConfig:e.trainingConfig!=null?e.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(a)),{modelArtifactsInfo:o}}catch{throw bi(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${o.modelTopologyBytes}, weightSpecsBytes=${o.weightSpecsBytes}, weightDataBytes=${o.weightDataBytes}.`)}}}async load(){const e=JSON.parse(this.LS.getItem(this.keys.info));if(e==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(e.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const t={},s=JSON.parse(this.LS.getItem(this.keys.topology));if(s==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);t.modelTopology=s;const o=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(o==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);t.weightSpecs=o;const r=this.LS.getItem(this.keys.modelMetadata);if(r!=null){const i=JSON.parse(r);t.format=i.format,t.generatedBy=i.generatedBy,t.convertedBy=i.convertedBy,i.signature!=null&&(t.signature=i.signature),i.userDefinedMetadata!=null&&(t.userDefinedMetadata=i.userDefinedMetadata),i.modelInitializer!=null&&(t.modelInitializer=i.modelInitializer),i.initializerSignature!=null&&(t.initializerSignature=i.initializerSignature),i.trainingConfig!=null&&(t.trainingConfig=i.trainingConfig)}const a=this.LS.getItem(this.keys.weightData);if(a==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return t.weightData=Yf(a),t}}vt.URL_SCHEME="localstorage://";const wi=n=>$().getBool("IS_BROWSER")&&!Array.isArray(n)&&n.startsWith(vt.URL_SCHEME)?dp(n.slice(vt.URL_SCHEME.length)):null;Y.registerSaveRouter(wi);Y.registerLoadRouter(wi);function dp(n){return new vt(n)}class hp{constructor(){R($().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),R(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const e={},t=Vt+Ke,s=Ke+gi;for(let o=0;o<this.LS.length;++o){const r=this.LS.key(o);if(r.startsWith(t)&&r.endsWith(s)){const a=lp(r);e[a]=JSON.parse(this.LS.getItem(r))}}return e}async removeModel(e){e=up(e);const t=xi(e);if(this.LS.getItem(t.info)==null)throw new Error(`Cannot find model at path '${e}'`);const s=JSON.parse(this.LS.getItem(t.info));return bi(t),s}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wt="://";class he{constructor(){this.managers={}}static getInstance(){return he.instance==null&&(he.instance=new he),he.instance}static registerManager(e,t){R(e!=null,()=>"scheme must not be undefined or null."),e.endsWith(Wt)&&(e=e.slice(0,e.indexOf(Wt))),R(e.length>0,()=>"scheme must not be an empty string.");const s=he.getInstance();R(s.managers[e]==null,()=>`A model store manager is already registered for scheme '${e}'.`),s.managers[e]=t}static getManager(e){const t=he.getInstance().managers[e];if(t==null)throw new Error(`Cannot find model manager for scheme '${e}'`);return t}static getSchemes(){return Object.keys(he.getInstance().managers)}}function Zn(n){if(n.indexOf(Wt)===-1)throw new Error(`The url string provided does not contain a scheme. Supported schemes are: ${he.getSchemes().join(",")}`);return{scheme:n.split(Wt)[0],path:n.split(Wt)[1]}}async function Ci(n,e,t=!1){R(n!==e,()=>`Old path and new path are the same: '${n}'`);const s=Y.getLoadHandlers(n);R(s.length>0,()=>`Copying failed because no load handler is found for source URL ${n}.`),R(s.length<2,()=>`Copying failed because more than one (${s.length}) load handlers for source URL ${n}.`);const o=s[0],r=Y.getSaveHandlers(e);R(r.length>0,()=>`Copying failed because no save handler is found for destination URL ${e}.`),R(r.length<2,()=>`Copying failed because more than one (${s.length}) save handlers for destination URL ${e}.`);const a=r[0],i=Zn(n).scheme,c=Zn(n).path,l=i===Zn(n).scheme,u=await o.load();t&&l&&await he.getManager(i).removeModel(c);const d=await a.save(u);return t&&!l&&await he.getManager(i).removeModel(c),d.modelArtifactsInfo}async function dR(){const n=he.getSchemes(),e={};for(const t of n){const s=await he.getManager(t).listModels();for(const o in s){const r=t+Wt+o;e[r]=s[o]}}return e}async function hR(n){const e=Zn(n);return he.getManager(e.scheme).removeModel(e.path)}async function fR(n,e){return Ci(n,e,!1)}async function pR(n,e){return Ci(n,e,!0)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class fp{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(e,t){return fetch(e,t)}now(){return performance.now()}encode(e,t){if(t!=="utf-8"&&t!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${t}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(e)}decode(e,t){return new TextDecoder(t).decode(e)}setTimeoutCustom(e,t){if(typeof window>"u"||!$().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(e,t);return}this.functionRefs.push(e),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},t),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",s=>{if(s.source===window&&s.data.name===this.messageName){s.stopPropagation();const o=this.functionRefs[s.data.index];o(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(e){return Ka(e)}}if($().get("IS_BROWSER")){$().setPlatform("browser",new fp);try{he.registerManager(vt.URL_SCHEME,new hp)}catch{}try{he.registerManager($t.URL_SCHEME,new op)}catch{}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pp={importFetch:()=>require("node-fetch")};let Bs;class mp{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(e,t){return $().global.fetch!=null?$().global.fetch(e,t):(Bs==null&&(Bs=pp.importFetch()),Bs(e,t))}now(){const e=process.hrtime();return e[0]*1e3+e[1]/1e6}encode(e,t){if(t!=="utf-8"&&t!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${t}`);return this.textEncoder.encode(e)}decode(e,t){return e.length===0?"":new this.util.TextDecoder(t).decode(e)}isTypedArray(e){return this.util.types.isFloat32Array(e)||this.util.types.isInt32Array(e)||this.util.types.isUint8Array(e)||this.util.types.isUint8ClampedArray(e)}}$().get("IS_NODE")&&!$().get("IS_BROWSER")&&$().setPlatform("node",new mp);/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ie(n,e="float32",t){return e=e||"float32",Qt(n),new is(n,e,t)}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gp(n,e){const t=M(n,"x","cast");if(!ya(e))throw new Error(`Failed to cast to unknown dtype ${e}`);if(e==="string"&&t.dtype!=="string"||e!=="string"&&t.dtype==="string")throw new Error("Only strings can be casted to strings");const s={x:t},o={dtype:e};return O.runKernel(gs,s,o)}const ls=Q({cast_:gp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xp(n){const t={x:M(n,"x","clone","string_or_numeric")};return O.runKernel(bs,t)}const yi=Q({clone_:xp});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bp(n,e=!1){console.log(n.toString(e))}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ai();const wp={buffer:ie,cast:ls,clone:yi,print:bp};Ef(wp);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cp(n,e){let t=M(n,"a","add"),s=M(e,"b","add");[t,s]=ut(t,s);const o={a:t,b:s};return O.runKernel(On,o)}const z=Q({add_:Cp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yp(n,e){let t=M(n,"a","floorDiv"),s=M(e,"b","floorDiv");[t,s]=ut(t,s);const o={a:t,b:s};return O.runKernel(xs,o)}const $p=Q({floorDiv_:yp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vp(n,e){let t=M(n,"a","div"),s=M(e,"b","div");if([t,s]=ut(t,s),t.dtype==="int32"&&s.dtype==="int32")return $p(t,s);const o={a:t,b:s},r={};return O.runKernel(Na,o,r)}const Me=Q({div_:vp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sp(n,e){let t=M(n,"a","mul"),s=M(e,"b","mul");[t,s]=ut(t,s);const o={a:t,b:s};return O.runKernel(Cs,o)}const V=Q({mul_:Sp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ip(n){const e=M(n,"x","abs");if(e.dtype==="complex64"){const t={x:e};return O.runKernel(ka,t)}else{const t={x:e};return O.runKernel(go,t)}}const Tp=Q({abs_:Ip});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $i(n,e,t,s,o="NHWC",r){const a=n[3],i=[...e,a],c=en(o);return _e(n,i,t,r,s,null,null,c)}function Zt(n,e,t,s,o,r,a="channelsLast"){const[i,c]=vn(e);let l;if(a==="channelsLast")l=[i,c,n[3],n[3]];else if(a==="channelsFirst")l=[i,c,n[1],n[1]];else throw new Error(`Unknown dataFormat ${a}`);return _e(n,l,t,s,o,r,!1,a)}function Ln(n,e,t,s,o,r,a="NDHWC"){const[i,c,l]=Zs(e);let u,d;if(a==="NDHWC")d="channelsLast",u=[i,c,l,n[4],n[4]];else if(a==="NCDHW")d="channelsFirst",u=[i,c,l,n[1],n[1]];else throw new Error(`Unknown dataFormat ${a}`);return Bn(n,u,t,s,o,!1,d,r)}function _e(n,e,t,s,o,r,a=!1,i="channelsLast"){let[c,l,u,d]=[-1,-1,-1,-1];if(i==="channelsLast")[c,l,u,d]=n;else if(i==="channelsFirst")[c,d,l,u]=n;else throw new Error(`Unknown dataFormat ${i}`);const[h,f,,p]=e,[x,g]=vn(t),[m,b]=vn(s),C=Gt(h,m),y=Gt(f,b),{padInfo:S,outHeight:N,outWidth:k}=kp(o,l,u,x,g,C,y,r,i),T=a?p*d:p;let I;return i==="channelsFirst"?I=[c,T,N,k]:i==="channelsLast"&&(I=[c,N,k,T]),{batchSize:c,dataFormat:i,inHeight:l,inWidth:u,inChannels:d,outHeight:N,outWidth:k,outChannels:T,padInfo:S,strideHeight:x,strideWidth:g,filterHeight:h,filterWidth:f,effectiveFilterHeight:C,effectiveFilterWidth:y,dilationHeight:m,dilationWidth:b,inShape:n,outShape:I,filterShape:e}}function Bn(n,e,t,s,o,r=!1,a="channelsLast",i){let[c,l,u,d,h]=[-1,-1,-1,-1,-1];if(a==="channelsLast")[c,l,u,d,h]=n;else if(a==="channelsFirst")[c,h,l,u,d]=n;else throw new Error(`Unknown dataFormat ${a}`);const[f,p,x,,g]=e,[m,b,C]=Zs(t),[y,S,N]=Zs(s),k=Gt(f,y),T=Gt(p,S),I=Gt(x,N),{padInfo:v,outDepth:w,outHeight:E,outWidth:_}=Np(o,l,u,d,m,b,C,k,T,I,i),B=r?g*h:g;let L;return a==="channelsFirst"?L=[c,B,w,E,_]:a==="channelsLast"&&(L=[c,w,E,_,B]),{batchSize:c,dataFormat:a,inDepth:l,inHeight:u,inWidth:d,inChannels:h,outDepth:w,outHeight:E,outWidth:_,outChannels:B,padInfo:v,strideDepth:m,strideHeight:b,strideWidth:C,filterDepth:f,filterHeight:p,filterWidth:x,effectiveFilterDepth:k,effectiveFilterHeight:T,effectiveFilterWidth:I,dilationDepth:y,dilationHeight:S,dilationWidth:N,inShape:n,outShape:L,filterShape:e}}function Rp(n,e,t,s,o){s==null&&(s=Vo(n,e,t));const r=n[0],a=n[1],i=Sn((r-e+2*s)/t+1,o),c=Sn((a-e+2*s)/t+1,o);return[i,c]}function Ep(n,e,t,s,o,r){o==null&&(o=Vo(n,e[0],s[0]));const a=[0,0,0,t];for(let i=0;i<3;i++)n[i]+2*o>=e[i]&&(a[i]=Sn((n[i]-e[i]+2*o)/s[i]+1,r));return a}function Vo(n,e,t,s=1){const o=Gt(e,s);return Math.floor((n[0]*(t-1)-t+o)/2)}function vn(n){return typeof n=="number"?[n,n,n]:n.length===2?[n[0],n[1],1]:n}function Zs(n){return typeof n=="number"?[n,n,n]:n}function Gt(n,e){return e<=1?n:n+(n-1)*(e-1)}function kp(n,e,t,s,o,r,a,i,c){let l,u,d;if(typeof n=="number"){l={top:n,bottom:n,left:n,right:n,type:n===0?"VALID":"NUMBER"};const f=Rp([e,t],r,s,n,i);u=f[0],d=f[1]}else if(n==="same"){u=Math.ceil(e/s),d=Math.ceil(t/o);const h=Math.max(0,(u-1)*s+r-e),f=Math.max(0,(d-1)*o+a-t),p=Math.floor(h/2),x=h-p,g=Math.floor(f/2),m=f-g;l={top:p,bottom:x,left:g,right:m,type:"SAME"}}else if(n==="valid")l={top:0,bottom:0,left:0,right:0,type:"VALID"},u=Math.ceil((e-r+1)/s),d=Math.ceil((t-a+1)/o);else if(typeof n=="object"){const h=c==="channelsLast"?n[1][0]:n[2][0],f=c==="channelsLast"?n[1][1]:n[2][1],p=c==="channelsLast"?n[2][0]:n[3][0],x=c==="channelsLast"?n[2][1]:n[3][1];l={top:h,bottom:f,left:p,right:x,type:h===0&&f===0&&p===0&&x===0?"VALID":"EXPLICIT"},u=Sn((e-r+h+f)/s+1,i),d=Sn((t-a+p+x)/o+1,i)}else throw Error(`Unknown padding parameter: ${n}`);return{padInfo:l,outHeight:u,outWidth:d}}function Np(n,e,t,s,o,r,a,i,c,l,u){let d,h,f,p;if(n==="valid"&&(n=0),typeof n=="number"){d={top:n,bottom:n,left:n,right:n,front:n,back:n,type:n===0?"VALID":"NUMBER"};const g=Ep([e,t,s,1],[i,c,l],1,[o,r,a],n,u);h=g[0],f=g[1],p=g[2]}else if(n==="same"){h=Math.ceil(e/o),f=Math.ceil(t/r),p=Math.ceil(s/a);const x=(h-1)*o+i-e,g=(f-1)*r+c-t,m=(p-1)*a+l-s,b=Math.floor(x/2),C=x-b,y=Math.floor(g/2),S=g-y,N=Math.floor(m/2),k=m-N;d={top:y,bottom:S,left:N,right:k,front:b,back:C,type:"SAME"}}else throw Error(`Unknown padding parameter: ${n}`);return{padInfo:d,outDepth:h,outHeight:f,outWidth:p}}function Sn(n,e){if(!e)return Math.trunc(n);switch(e){case"round":return Math.round(n);case"ceil":return Math.ceil(n);case"floor":return Math.floor(n);default:throw new Error(`Unknown roundingMode ${e}`)}}function Js(n){const[e,t,s]=vn(n);return e===1&&t===1&&s===1}function Jt(n,e){return Js(n)||Js(e)}function Ap(n){return vn(n).every(e=>e>0)}function en(n){if(n==="NHWC")return"channelsLast";if(n==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${n}`)}function Fp(n,e,t){if(t!=null){if(typeof e=="string")throw Error(`Error in ${n}: pad must be an integer when using dimRoundingMode ${t} but got pad ${e}.`);if(typeof e=="number")R(Cn(e),()=>`Error in ${n}: pad must be an integer when using dimRoundingMode ${t} but got pad ${e}.`);else if(typeof e=="object")e.forEach(s=>{s.forEach(o=>{R(Cn(o),()=>`Error in ${n}: pad must be an integer when using dimRoundingMode ${t} but got pad ${o}.`)})});else throw Error(`Error in ${n}: Unknown padding parameter: ${e}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dp(n,e){const s={x:M(n,"x","reshape","string_or_numeric")},o={shape:e};return O.runKernel(Va,s,o)}const Uo=Q({reshape_:Dp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Op(n){const t={x:M(n,"x","sigmoid","float32")};return O.runKernel(ys,t)}const Pp=Q({sigmoid_:Op});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _p(n,e){let t=M(n,"broadcastTo","x");const s=t.shape;if(Qt(e),e.length<t.rank)throw new Error(`broadcastTo(): shape.length=${e.length} < input.rank=${t.rank}.`);if(e.length>t.rank){const l=t.shape.slice();for(;l.length<e.length;)l.unshift(1);t=Uo(t,l)}const o=t.shape,r=Array.from(e);for(let l=e.length-1;l>=0;l--)if(o[l]===e[l])r[l]=1;else if(t.shape[l]!==1)throw new Error(`broadcastTo(): [${s}] cannot be broadcast to [${e}].`);if(r.map((l,u)=>l>1?u:-1).filter(l=>l>=0).length===0)return yi(t);const i={x:t},c={reps:r};return O.runKernel(za,i,c)}const Lp=Q({broadcastTo_:_p});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bp(n,e,t){Qt(n),t=t||Yt(e);const s={shape:n,value:e,dtype:t};return O.runKernel(Fa,{},s)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function St(n,e){const t=n.length,s=[];for(let o=0;o<t;o++){const r=t-1-o,a=n[r]||1;(e[e.length-1-o]||1)>1&&a===1&&s.unshift(r)}return s}function Wo(n,e){const t=[];for(let s=0;s<e.length;s++){const o=n[n.length-s-1],r=e.length-s-1,a=e[r];(o==null||o===1&&a>1)&&t.unshift(r)}return t}function ae(n,e){const t=Math.max(n.length,e.length),s=new Array(t);for(let o=0;o<t;o++){let r=n[n.length-o-1];r==null&&(r=1);let a=e[e.length-o-1];if(a==null&&(a=1),r===1)s[t-o-1]=a;else if(a===1)s[t-o-1]=r;else if(r!==a){const i=`Operands could not be broadcast together with shapes ${n} and ${e}.`;throw Error(i)}else s[t-o-1]=r}return s}const mR=Object.freeze(Object.defineProperty({__proto__:null,assertAndGetBroadcastShape:ae,getBroadcastDims:St,getReductionAxes:Wo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mp(n){const t={x:M(n,"x","zerosLike")};return O.runKernel(Xa,t)}const Ve=Q({zerosLike_:Mp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vp(n){const t={x:M(n,"x","elu","float32")};return O.runKernel(Aa,t)}const Up=Q({elu_:Vp});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Go(n,e){for(let t=0;t<n.length;++t)if(n[n.length-t-1]!==e-1-t)return!1;return!0}function vi(n,e,t){const s=n.length+e.length,o=[];let r=0,a=0;for(let i=0;i<s;i++)t.indexOf(i)===-1?o.push(n[r++]):o.push(e[a++]);return o}function He(n,e){const t=[],s=n.length;for(let r=0;r<s;r++)e.indexOf(r)===-1&&t.push(n[r]);const o=e.map(r=>n[r]);return[t,o]}function Xe(n,e){const t=e.map(s=>1);return vi(n,t,e)}function Le(n,e,t){R(Go(e,t),()=>`${n} supports only inner-most axes for now. Got axes ${e} and rank-${t} input.`)}function Re(n,e){if(Go(n,e))return null;const t=[];for(let s=0;s<e;++s)n.indexOf(s)===-1&&t.push(s);return n.forEach(s=>t.push(s)),t}function zo(n){return n.map((e,t)=>[t,e]).sort((e,t)=>e[1]-t[1]).map(e=>e[0])}function Ee(n,e){const t=[];for(let s=e-n;s<e;++s)t.push(s);return t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wp(n,e){let t=M(n,"base","pow"),s=M(e,"exp","pow");[t,s]=ut(t,s);const o={a:t,b:s};return O.runKernel(Pa,o)}const Or=Q({pow_:Wp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ct(n,e){if((Te(n)&&e!=="string"||Array.isArray(n))&&e!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(e==="string"&&Te(n)&&!(n instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return ui(n,[],[],e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gp(n){const t={x:M(n,"x","sqrt","float32")};return O.runKernel($s,t)}const jt=Q({sqrt_:Gp});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zp(n){const e=M(n,"x","square"),t={};return O.runKernel("Square",{x:e},t)}const wt=Q({square_:zp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hp(n,e=null,t=!1){let s=M(n,"x","sum");s.dtype==="bool"&&(s=ls(s,"int32"));const o={x:s},r={axis:e,keepDims:t};return O.runKernel(Ga,o,r)}const Xp=Q({sum_:Hp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qp(n,e=.2){const s={x:M(n,"x","leakyRelu")},o={alpha:e};return O.runKernel(Da,s,o)}const jp=Q({leakyRelu_:qp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gR(n){return R(rt(n),()=>"The f passed in grad(f) must be a function"),(e,t)=>{const s=M(e,"x","tf.grad","string_or_numeric"),o=t!=null?M(t,"dy","tf.grad"):null;return O.tidy(()=>{const{value:r,grads:a}=O.gradients(()=>n(s),[s],o);return o!=null&&Kt(r.shape,o.shape,"The shape of dy passed in grad(f)(x, dy) must match the shape returned by f(x)"),Is(a),a[0]})}}function xR(n){return R(rt(n),()=>"The f passed in grads(f) must be a function"),(e,t)=>{R(Array.isArray(e),()=>"The args passed in grads(f)(args) must be an array of `Tensor`s or `TensorLike`s");const s=Lf(e,"args","tf.grads","string_or_numeric"),o=t!=null?M(t,"dy","tf.grads"):null;return O.tidy(()=>{const{value:r,grads:a}=O.gradients(()=>n(...s),s,o);return o!=null&&Kt(r.shape,o.shape,"The shape of dy passed in grads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),Is(a),a})}}function bR(n){return R(rt(n),()=>"The f passed in valueAndGrad(f) must be a function"),(e,t)=>{R(e instanceof xe,()=>"The x passed in valueAndGrad(f)(x) must be a tensor"),R(t==null||t instanceof xe,()=>"The dy passed in valueAndGrad(f)(x, dy) must be a tensor");const{grads:s,value:o}=O.gradients(()=>n(e),[e],t);return Is(s),{grad:s[0],value:o}}}function wR(n){return R(rt(n),()=>"The f passed in valueAndGrads(f) must be a function"),(e,t)=>{R(Array.isArray(e)&&e.every(o=>o instanceof xe),()=>"The args passed in valueAndGrads(f)(args) must be array of tensors"),R(t==null||t instanceof xe,()=>"The dy passed in valueAndGrads(f)(args, dy) must be a tensor");const s=O.gradients(()=>n(...e),e,t);return t!=null&&Kt(s.value.shape,t.shape,"The shape of dy passed in valueAndGrads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),Is(s.grads),s}}function Kp(n,e){R(rt(n),()=>"The f passed in variableGrads(f) must be a function"),R(e==null||Array.isArray(e)&&e.every(l=>l instanceof cs),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const t=e!=null;if(!t){e=[];for(const l in O.registeredVariables)e.push(O.registeredVariables[l])}const s=t?e.filter(l=>!l.trainable):null,o=e.length;e=e.filter(l=>l.trainable),R(e.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${o} variables is trainable.`);const r=!0,{value:a,grads:i}=O.gradients(n,e,null,r);R(i.some(l=>l!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),R(a.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${a.rank} tensor`);const c={};return e.forEach((l,u)=>{i[u]!=null&&(c[l.name]=i[u])}),s!=null&&s.forEach(l=>c[l.name]=null),{value:a,grads:c}}function CR(n){return O.customGrad(n)}function Is(n){if(n.filter(t=>t==null).length>0)throw new Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that
    the f you passed encloses all operations that lead from x to y.`)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yp(n,e){let t=M(n,"a","sub"),s=M(e,"b","sub");[t,s]=ut(t,s);const o={a:t,b:s};return O.runKernel(vs,o)}const zt=Q({sub_:Yp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qp(n,e){let t=M(n,"a","maximum"),s=M(e,"b","maximum");[t,s]=ut(t,s),t.dtype==="bool"&&(t=ls(t,"int32"),s=ls(s,"int32")),ae(t.shape,s.shape);const o={a:t,b:s};return O.runKernel(ws,o)}const Zp=Q({maximum_:Qp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eo(n,e="float32"){if(Qt(n),e==="complex64"){const s=eo(n,"float32"),o=eo(n,"float32");return li(s,o)}const t=We(A(n),e);return O.makeTensor(t,n,e)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jp(n,e){const t=M(n,"x","prelu"),s=M(e,"alpha","prelu"),o={x:t,alpha:s};return O.runKernel(_a,o)}const em=Q({prelu_:Jp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tm(n){const t={x:M(n,"x","relu")};return O.runKernel(Ma,t)}const nm=Q({relu_:tm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sm(n){const t={x:M(n,"x","relu6")};return O.runKernel(Ua,t)}const om=Q({relu6_:sm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rm(n,e=0){const s={x:M(n,"x","step")},o={alpha:e};return O.runKernel(qa,s,o)}const am=Q({step_:rm});function Ho(n,e,t){const s=e.rank>1?e.shape[e.rank-1]:1,o=e.rank>1?e.rank-1:1,r=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${t.shape}, indices.shape: ${e.shape}, shape: ${n}, sliceDim: ${s}, and batchDim: ${o}.`;if(t.rank<o)throw new Error(r+` update.rank < ${o}. `);if(n.length<s+(t.rank-o))throw new Error(r+` Output shape length < ${s+(t.rank-o)}`);if(t.rank!==o+n.length-s)throw new Error(r+` update.rank != ${o+n.length-s}`);for(let a=0;a<o;++a)if(t.shape[a]!==e.shape[a])throw new Error(r+` updates.shape[${a}] (${t.shape[a]}) != indices.shape[${a}] (${e.shape[a]}).`);for(let a=0;a<t.rank-o;++a)if(t.shape[a+o]!==n[a+s])throw new Error(r+` updates.shape[${a+o}] (${t.shape[a+o]}) != shape[${a+o}] (${n[a+o]})`)}function Si(n,e,t){if(e.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${e.rank}.`);if(n.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${n.rank}.`);if(e.dtype!=="int32")throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${e.dtype}`);if(t.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${t}`);if(t.length===0){if(e.size===0)throw new Error(`Indices specified for empty output. indices shape: ${e.shape}`);if(n.size===0)throw new Error(`Updates specified for empty output. updates shape: ${n.shape}`)}Ho(t,e,n)}function Mn(n,e,t){const s=e.shape.length,o=s>1?e.shape[s-1]:1,r=t.length;let a=1;for(let d=o;d<r;++d)a*=t[d];const i=o<1?1:o,c=A(e.shape)/i,l=[...J(t.slice(0,o)),1],u=A(t);return{sliceRank:o,numUpdates:c,sliceSize:a,strides:l,outputSize:u}}const yR=Object.freeze(Object.defineProperty({__proto__:null,calculateShapes:Mn,validateInput:Si,validateUpdateShape:Ho},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function im(n,e){const t=[];for(let r=0;r<e.length;r++)e[r]&&t.push(r);const s=ie(n,"int32"),o=ie([t.length,n.length],"int32");for(let r=0;r<t.length;r++){const a=s.indexToLoc(t[r]),i=r*n.length;o.values.set(a,i)}return o.toTensor()}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cm(n,e,t){if(t==null||t==="linear")return n;if(t==="relu")return V(n,am(e));throw new Error(`Cannot compute gradient for fused activation ${t}.`)}function lm(n,e){let t=e;const s=Wo(n.shape,e.shape);return s.length>0&&(t=Xp(t,s)),Uo(t,n.shape)}function um(n,e,t,s){if(e==="linear")return n;if(e==="relu")return nm(n);if(e==="elu")return Up(n);if(e==="relu6")return om(n);if(e==="prelu")return em(n,t);if(e==="leakyrelu")return jp(n,s);if(e==="sigmoid")return Pp(n);throw new Error(`Unknown fused activation ${e}.`)}const dm=(n,e)=>!(n>0)||e==="linear";/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hm(n,e,t){const s=fm(n,e,t),o=s<0?-(s+1):s;n.splice(o,0,e)}function fm(n,e,t){return mm(n,e,t||pm)}function pm(n,e){return n>e?1:n<e?-1:0}function mm(n,e,t){let s=0,o=n.length,r=0,a=!1;for(;s<o;){r=s+(o-s>>>1);const i=t(e,n[r]);i>0?s=r+1:(o=r,a=!i)}return a?s:-s-1}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gm(n,e,t,s,o){return Xo(n,e,t,s,o,0)}function xm(n,e,t,s,o,r){return Xo(n,e,t,s,o,0,!1,r,!0)}function bm(n,e,t,s,o,r){return Xo(n,e,t,s,o,r,!0)}function Xo(n,e,t,s,o,r,a=!1,i=!1,c=!1){const l=[];for(let g=0;g<e.length;g++)e[g]>o&&l.push({score:e[g],boxIndex:g,suppressBeginIndex:0});l.sort(Pr);const u=r>0?-.5/r:0,d=[],h=[];for(;d.length<t&&l.length>0;){const g=l.pop(),{score:m,boxIndex:b,suppressBeginIndex:C}=g;if(m<o)break;let y=!1;for(let S=d.length-1;S>=C;--S){const N=wm(n,b,d[S]);if(N>=s){y=!0;break}if(g.score=g.score*Cm(s,u,N),g.score<=o)break}g.suppressBeginIndex=d.length,y||(g.score===m?(d.push(b),h.push(g.score)):g.score>o&&hm(l,g,Pr))}const f=d.length,p=t-f;i&&p>0&&(d.push(...new Array(p).fill(0)),h.push(...new Array(p).fill(0)));const x={selectedIndices:d};return a&&(x.selectedScores=h),c&&(x.validOutputs=f),x}function wm(n,e,t){const s=n.subarray(e*4,e*4+4),o=n.subarray(t*4,t*4+4),r=Math.min(s[0],s[2]),a=Math.min(s[1],s[3]),i=Math.max(s[0],s[2]),c=Math.max(s[1],s[3]),l=Math.min(o[0],o[2]),u=Math.min(o[1],o[3]),d=Math.max(o[0],o[2]),h=Math.max(o[1],o[3]),f=(i-r)*(c-a),p=(d-l)*(h-u);if(f<=0||p<=0)return 0;const x=Math.max(r,l),g=Math.max(a,u),m=Math.min(i,d),b=Math.min(c,h),C=Math.max(m-x,0)*Math.max(b-g,0);return C/(f+p-C)}function Cm(n,e,t){const s=Math.exp(e*t*t);return t<=n?s:0}function Pr(n,e){return n.score-e.score||n.score===e.score&&e.boxIndex-n.boxIndex}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ym=new Map,to=new Map;class Ii{getClassName(){return this.constructor.className}static fromConfig(e,t){return new e(t)}}class et{constructor(){this.classNameMap={}}static getMap(){return et.instance==null&&(et.instance=new et),et.instance}static register(e){et.getMap().classNameMap[e.className]=[e,e.fromConfig]}}function Ti(n,e,t){R(n.className!=null,()=>"Class being registered does not have the static className property defined."),R(typeof n.className=="string",()=>"className is required to be a string, but got type "+typeof n.className),R(n.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),typeof e>"u"&&(e="Custom"),typeof t>"u"&&(t=n.className);const s=t,o=e+">"+s;return et.register(n),ym.set(o,n),to.set(n,o),n}function $m(n){return to.has(n)?to.get(n):n.className}const $R=Object.freeze(Object.defineProperty({__proto__:null,Serializable:Ii,SerializationMap:et,getRegisteredName:$m,registerClass:Ti},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class At extends Ii{minimize(e,t=!1,s){const{value:o,grads:r}=this.computeGradients(e,s);if(s!=null){const a=s.map(i=>({name:i.name,tensor:r[i.name]}));this.applyGradients(a)}else this.applyGradients(r);return Ie(r),t?o:(o.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(e,t){return Kp(e,t)}dispose(){this.iterations_!=null&&Ie(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:ct(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(e){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(e){return this.iterations_=(await e[0].tensor.data())[0],e.slice(1)}}Object.defineProperty(At,Symbol.hasInstance,{value:n=>n.minimize!=null&&n.computeGradients!=null&&n.applyGradients!=null});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class vm extends At{static get className(){return"Adadelta"}constructor(e,t,s=null){super(),this.learningRate=e,this.rho=t,this.epsilon=s,this.accumulatedGrads=[],this.accumulatedUpdates=[],s==null&&(this.epsilon=O.backend.epsilon())}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,o)=>{const r=O.registeredVariables[s],a=!1;this.accumulatedGrads[o]==null&&(this.accumulatedGrads[o]={originalName:`${s}/accum_grad`,variable:ee(()=>Ve(r).variable(a))}),this.accumulatedUpdates[o]==null&&(this.accumulatedUpdates[o]={originalName:`${s}/accum_var`,variable:ee(()=>Ve(r).variable(a))});const i=Array.isArray(e)?e[o].tensor:e[s];if(i==null)return;const c=this.accumulatedGrads[o].variable,l=this.accumulatedUpdates[o].variable;ee(()=>{const u=z(V(c,this.rho),V(wt(i),1-this.rho)),d=V(Me(jt(z(l,this.epsilon)),jt(z(c,this.epsilon))),i),h=z(V(l,this.rho),V(wt(d),1-this.rho));c.assign(u),l.assign(h);const f=z(V(d,-this.learningRate),r);r.assign(f)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(Ie(this.accumulatedGrads.map(e=>e.variable)),Ie(this.accumulatedUpdates.map(e=>e.variable)))}async getWeights(){const e=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(e.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=e.length/2,s=!1;this.accumulatedGrads=e.slice(0,t).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})),this.accumulatedUpdates=e.slice(t,t*2).map(o=>({originalName:o.name,variable:o.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.rho,t.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Sm extends At{static get className(){return"Adagrad"}constructor(e,t=.1){super(),this.learningRate=e,this.initialAccumulatorValue=t,this.accumulatedGrads=[]}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,o)=>{const r=O.registeredVariables[s];this.accumulatedGrads[o]==null&&(this.accumulatedGrads[o]={originalName:`${s}/accumulator`,variable:ee(()=>Bp(r.shape,this.initialAccumulatorValue).variable(!1))});const a=Array.isArray(e)?e[o].tensor:e[s];if(a==null)return;const i=this.accumulatedGrads[o].variable;ee(()=>{const c=z(i,wt(a));i.assign(c);const l=z(V(Me(a,jt(z(c,O.backend.epsilon()))),-this.learningRate),r);r.assign(l)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&Ie(this.accumulatedGrads.map(e=>e.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=!1;this.accumulatedGrads=e.map(s=>({originalName:s.name,variable:s.tensor.variable(t)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(e,t){return new e(t.learningRate,t.initialAccumulatorValue)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Im extends At{static get className(){return"Adam"}constructor(e,t,s,o=null){super(),this.learningRate=e,this.beta1=t,this.beta2=s,this.epsilon=o,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],ee(()=>{this.accBeta1=ct(t).variable(),this.accBeta2=ct(s).variable()}),o==null&&(this.epsilon=O.backend.epsilon())}applyGradients(e){const t=Array.isArray(e)?e.map(s=>s.name):Object.keys(e);ee(()=>{const s=zt(1,this.accBeta1),o=zt(1,this.accBeta2);t.forEach((r,a)=>{const i=O.registeredVariables[r],c=!1;this.accumulatedFirstMoment[a]==null&&(this.accumulatedFirstMoment[a]={originalName:`${r}/m`,variable:ee(()=>Ve(i).variable(c))}),this.accumulatedSecondMoment[a]==null&&(this.accumulatedSecondMoment[a]={originalName:`${r}/v`,variable:ee(()=>Ve(i).variable(c))});const l=Array.isArray(e)?e[a].tensor:e[r];if(l==null)return;const u=this.accumulatedFirstMoment[a].variable,d=this.accumulatedSecondMoment[a].variable,h=z(V(u,this.beta1),V(l,1-this.beta1)),f=z(V(d,this.beta2),V(wt(l),1-this.beta2)),p=Me(h,s),x=Me(f,o);u.assign(h),d.assign(f);const g=z(V(Me(p,z(jt(x),this.epsilon)),-this.learningRate),i);i.assign(g)}),this.accBeta1.assign(V(this.accBeta1,this.beta1)),this.accBeta2.assign(V(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&Ie(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedSecondMoment!=null&&Ie(this.accumulatedSecondMoment.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(e.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(e){e=await this.extractIterations(e),ee(()=>{this.accBeta1.assign(Or(this.beta1,this.iterations_+1)),this.accBeta2.assign(Or(this.beta2,this.iterations_+1))});const t=e.length/2,s=!1;this.accumulatedFirstMoment=e.slice(0,t).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})),this.accumulatedSecondMoment=e.slice(t,t*2).map(o=>({originalName:o.name,variable:o.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Tm extends At{static get className(){return"Adamax"}constructor(e,t,s,o=null,r=0){super(),this.learningRate=e,this.beta1=t,this.beta2=s,this.epsilon=o,this.decay=r,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],ee(()=>{this.iteration=ct(0).variable(),this.accBeta1=ct(t).variable()}),o==null&&(this.epsilon=O.backend.epsilon())}applyGradients(e){const t=Array.isArray(e)?e.map(s=>s.name):Object.keys(e);ee(()=>{const s=zt(1,this.accBeta1),o=Me(-this.learningRate,z(V(this.iteration,this.decay),1));t.forEach((r,a)=>{const i=O.registeredVariables[r],c=!1;this.accumulatedFirstMoment[a]==null&&(this.accumulatedFirstMoment[a]={originalName:`${r}/m`,variable:Ve(i).variable(c)}),this.accumulatedWeightedInfNorm[a]==null&&(this.accumulatedWeightedInfNorm[a]={originalName:`${r}/v`,variable:Ve(i).variable(c)});const l=Array.isArray(e)?e[a].tensor:e[r];if(l==null)return;const u=this.accumulatedFirstMoment[a].variable,d=this.accumulatedWeightedInfNorm[a].variable,h=z(V(u,this.beta1),V(l,1-this.beta1)),f=V(d,this.beta2),p=Tp(l),x=Zp(f,p);u.assign(h),d.assign(x);const g=z(V(Me(o,s),Me(h,z(x,this.epsilon))),i);i.assign(g)}),this.iteration.assign(z(this.iteration,1)),this.accBeta1.assign(V(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&Ie(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedWeightedInfNorm!=null&&Ie(this.accumulatedWeightedInfNorm.map(e=>e.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(e){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon,t.decay)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ri extends At{static get className(){return"SGD"}constructor(e){super(),this.learningRate=e,this.setLearningRate(e)}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,o)=>{const r=Array.isArray(e)?e[o].tensor:e[s];if(r==null)return;const a=O.registeredVariables[s];ee(()=>{const i=z(V(this.c,r),a);a.assign(i)})}),this.incrementIterations()}setLearningRate(e){this.learningRate=e,this.c!=null&&this.c.dispose(),this.c=Uf(ct(-e))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(e){if(e=await this.extractIterations(e),e.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(e,t){return new e(t.learningRate)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Rm extends Ri{static get className(){return"Momentum"}constructor(e,t,s=!1){super(e),this.learningRate=e,this.momentum=t,this.useNesterov=s,this.accumulations=[],this.m=ct(this.momentum)}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,o)=>{const r=O.registeredVariables[s];this.accumulations[o]==null&&(this.accumulations[o]={originalName:`${s}/momentum`,variable:ee(()=>Ve(r).variable(!1))});const a=this.accumulations[o].variable,i=Array.isArray(e)?e[o].tensor:e[s];i!=null&&ee(()=>{let c;const l=z(V(this.m,a),i);this.useNesterov?c=z(V(this.c,z(i,V(l,this.m))),r):c=z(V(this.c,l),r),a.assign(l),r.assign(c)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&Ie(this.accumulations.map(e=>e.variable))}setMomentum(e){this.momentum=e}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=!1;this.accumulations=e.map(s=>({originalName:s.name,variable:s.tensor.variable(t)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(e,t){return new e(t.learningRate,t.momentum,t.useNesterov)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Em extends At{static get className(){return"RMSProp"}constructor(e,t=.9,s=0,o=null,r=!1){if(super(),this.learningRate=e,this.decay=t,this.momentum=s,this.epsilon=o,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=r,o==null&&(this.epsilon=O.backend.epsilon()),e==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,o)=>{const r=O.registeredVariables[s],a=!1;this.accumulatedMeanSquares[o]==null&&(this.accumulatedMeanSquares[o]={originalName:`${s}/rms`,variable:ee(()=>Ve(r).variable(a))}),this.accumulatedMoments[o]==null&&(this.accumulatedMoments[o]={originalName:`${s}/momentum`,variable:ee(()=>Ve(r).variable(a))}),this.accumulatedMeanGrads[o]==null&&this.centered&&(this.accumulatedMeanGrads[o]={originalName:`${s}/mg`,variable:ee(()=>Ve(r).variable(a))});const i=Array.isArray(e)?e[o].tensor:e[s];if(i==null)return;const c=this.accumulatedMeanSquares[o].variable,l=this.accumulatedMoments[o].variable;ee(()=>{const u=z(V(c,this.decay),V(wt(i),1-this.decay));if(this.centered){const d=this.accumulatedMeanGrads[o].variable,h=z(V(d,this.decay),V(i,1-this.decay)),f=Me(V(i,this.learningRate),jt(zt(u,z(wt(h),this.epsilon)))),p=z(V(l,this.momentum),f);c.assign(u),d.assign(h),l.assign(p);const x=zt(r,p);r.assign(x)}else{const d=z(V(c,this.decay),V(wt(i),1-this.decay)),h=z(V(l,this.momentum),Me(V(i,this.learningRate),jt(z(d,this.epsilon))));c.assign(d),l.assign(h);const f=zt(r,h);r.assign(f)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&Ie(this.accumulatedMeanSquares.map(e=>e.variable)),this.accumulatedMeanGrads!=null&&this.centered&&Ie(this.accumulatedMeanGrads.map(e=>e.variable)),this.accumulatedMoments!=null&&Ie(this.accumulatedMoments.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&e.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(e.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=this.centered?e.length/3:e.length/2,s=!1;this.accumulatedMeanSquares=e.slice(0,t).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})),this.accumulatedMoments=e.slice(t,t*2).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})),this.centered&&(this.accumulatedMeanGrads=e.slice(t*2,t*3).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(e,t){return new e(t.learningRate,t.decay,t.momentum,t.epsilon,t.centered)}}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const km=[vm,Sm,Im,Tm,Rm,Em,Ri];function Nm(){for(const n of km)Ti(n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Am="model",Fm=".json",Dm=".weights.bin";function _r(n){return new Promise(e=>setTimeout(e)).then(n)}class It{constructor(e){if(!$().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");e.startsWith(It.URL_SCHEME)&&(e=e.slice(It.URL_SCHEME.length)),(e==null||e.length===0)&&(e=Am),this.modelJsonFileName=e+Fm,this.weightDataFileName=e+Dm}async save(e){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const t=ze.join(e.weightData),s=window.URL.createObjectURL(new Blob([t],{type:"application/octet-stream"}));if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const o=[{paths:["./"+this.weightDataFileName],weights:e.weightSpecs}],r=hi(e,o),a=window.URL.createObjectURL(new Blob([JSON.stringify(r)],{type:"application/json"})),i=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(i.download=this.modelJsonFileName,i.href=a,await _r(()=>i.dispatchEvent(new MouseEvent("click"))),e.weightData!=null){const c=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;c.download=this.weightDataFileName,c.href=s,await _r(()=>c.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:Ss(e)}}}}It.URL_SCHEME="downloads://";class Om{constructor(e){if(e==null||e.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${e}`);this.jsonFile=e[0],this.weightsFiles=e.slice(1)}async load(){return new Promise((e,t)=>{const s=new FileReader;s.onload=o=>{const r=JSON.parse(o.target.result),a=r.modelTopology;if(a==null){t(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(r.weightsManifest==null){t(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){e({modelTopology:a});return}const c=fi(r,l=>this.loadWeights(l));e(c)},s.onerror=o=>t(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),s.readAsText(this.jsonFile)})}loadWeights(e){const t=[],s=[];for(const a of e)t.push(...a.weights),s.push(...a.paths);const o=this.checkManifestAndWeightFiles(e),r=s.map(a=>this.loadWeightsFile(a,o[a]));return Promise.all(r).then(a=>[t,a])}loadWeightsFile(e,t){return new Promise((s,o)=>{const r=new FileReader;r.onload=a=>{const i=a.target.result;s(i)},r.onerror=a=>o(`Failed to weights data from file of path '${e}'.`),r.readAsArrayBuffer(t)})}checkManifestAndWeightFiles(e){const t=[],s=this.weightsFiles.map(r=>Fr(r.name)),o={};for(const r of e)r.paths.forEach(a=>{const i=Fr(a);if(t.indexOf(i)!==-1)throw new Error(`Duplicate file basename found in weights manifest: '${i}'`);if(t.push(i),s.indexOf(i)===-1)throw new Error(`Weight file with basename '${i}' is not provided.`);o[a]=this.weightsFiles[s.indexOf(i)]});if(t.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${t.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return o}}const Pm=n=>$().getBool("IS_BROWSER")&&!Array.isArray(n)&&n.startsWith(It.URL_SCHEME)?_m(n.slice(It.URL_SCHEME.length)):null;Y.registerSaveRouter(Pm);function _m(n="model"){return new It(n)}function vR(n){return new Om(n)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lr(n,e,t,s){a(n),t=t??0,s=s??1,i(t,s);let o=0;const r=c=>(c.then(l=>{const u=t+ ++o/n.length*(s-t);return e(u),l}),c);function a(c){R(c!=null&&Array.isArray(c)&&c.length>0,()=>"promises must be a none empty array")}function i(c,l){R(c>=0&&c<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${c}`),R(l>=0&&l<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${l}`),R(l>=c,()=>`startFraction must be no more than endFraction, but got startFraction ${c} and endFraction ${l}`)}return Promise.all(n.map(r))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Ei(n,e){e==null&&(e={});const t=e.fetchFunc==null?$().platform.fetch:e.fetchFunc,s=n.map(d=>t(d,e.requestInit,{isBinary:!0})),i=(e.onProgress==null?await Promise.all(s):await Lr(s,e.onProgress,0,.5)).map(d=>d.arrayBuffer());return e.onProgress==null?await Promise.all(i):await Lr(i,e.onProgress,.5,1)}function Lm(n,e){var t;const s=e.fetchFunc==null?$().platform.fetch:e.fetchFunc;let o=0,r;return(t=e.onProgress)===null||t===void 0||t.call(e,0),new ReadableStream({pull:async a=>{for(var i;o<n.length;){r||(r=(await s(n[o],e.requestInit,{isBinary:!0})).body.getReader());const{done:c,value:l}=await r.read();if(c){o++,r=void 0,(i=e.onProgress)===null||i===void 0||i.call(e,o/n.length);continue}a.enqueue(l);return}a.close()}})}async function SR(n,e="",t,s){return Bm(a=>Ei(a,{requestInit:s}))(n,e,t)}function Bm(n){return async(e,t="",s)=>{const o=e.map(()=>!1),r={},a=s!=null?s.map(()=>!1):[],i=[];if(e.forEach((f,p)=>{let x=0;f.weights.forEach(g=>{const m="quantization"in g?g.quantization.dtype:g.dtype,b=yt[m]*A(g.shape),C=()=>{o[p]=!0,r[p]==null&&(r[p]=[]),r[p].push({manifestEntry:g,groupOffset:x,sizeBytes:b})};s!=null?s.forEach((y,S)=>{y===g.name&&(C(),a[S]=!0)}):C(),i.push(g.name),x+=b})}),!a.every(f=>f)){const f=s.filter((p,x)=>!a[x]);throw new Error(`Could not find weights in manifest with names: ${f.join(", ")}. 
Manifest JSON has weights with names: ${i.join(", ")}.`)}const c=o.reduce((f,p,x)=>(p&&f.push(x),f),[]),l=[];c.forEach(f=>{e[f].paths.forEach(p=>{const x=t+(t.endsWith("/")?"":"/")+p;l.push(x)})});const u=await n(l),d={};let h=0;return c.forEach(f=>{const p=e[f].paths.length,x=new ze(u.slice(h,h+p));r[f].forEach(m=>{const b=x.slice(m.groupOffset,m.groupOffset+m.sizeBytes),C=Hf(b,[m.manifestEntry]);for(const y in C)d[y]=C[y]}),h+=p}),d}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mm="application/octet-stream",Vm="application/json";class qo{constructor(e,t){if(this.DEFAULT_METHOD="POST",t==null&&(t={}),this.weightPathPrefix=t.weightPathPrefix,this.weightUrlConverter=t.weightUrlConverter,t.fetchFunc!=null?(R(typeof t.fetchFunc=="function",()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=t.fetchFunc):this.fetch=$().platform.fetch,R(e!=null&&e.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(e)&&R(e.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${e.length}).`),this.path=e,t.requestInit!=null&&t.requestInit.body!=null)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=t.requestInit||{},this.loadOptions=t}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const t=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);t.body=new FormData;const s=[{paths:["./model.weights.bin"],weights:e.weightSpecs}],o=hi(e,s);if(t.body.append("model.json",new Blob([JSON.stringify(o)],{type:Vm}),"model.json"),e.weightData!=null){const a=ze.join(e.weightData);t.body.append("model.weights.bin",new Blob([a],{type:Mm}),"model.weights.bin")}const r=await this.fetch(this.path,t);if(r.ok)return{modelArtifactsInfo:Ss(e),responses:[r]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${r.status}.`)}async loadModelJSON(){const e=await this.fetch(this.path,this.requestInit);if(!e.ok)throw new Error(`Request to ${this.path} failed with status code ${e.status}. Please verify this URL points to the model JSON of the model to load.`);let t;try{t=await e.json()}catch{let a=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?a+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":a+=" Please make sure the server is serving valid JSON for this request.",new Error(a)}const s=t.modelTopology,o=t.weightsManifest;if(s==null&&o==null)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return t}async load(){if(this.loadOptions.streamWeights)return this.loadStream();const e=await this.loadModelJSON();return fi(e,t=>this.loadWeights(t))}async loadStream(){const e=await this.loadModelJSON(),t=await this.getWeightUrls(e.weightsManifest),s=Dr(e.weightsManifest),o=()=>Lm(t,this.loadOptions);return Object.assign(Object.assign({},e),{weightSpecs:s,getWeightStream:o})}async getWeightUrls(e){const t=Array.isArray(this.path)?this.path[1]:this.path,[s,o]=Um(t),r=this.weightPathPrefix||s,a=[],i=[];for(const c of e)for(const l of c.paths)this.weightUrlConverter!=null?i.push(this.weightUrlConverter(l)):a.push(r+l+o);return this.weightUrlConverter&&a.push(...await Promise.all(i)),a}async loadWeights(e){const t=await this.getWeightUrls(e),s=Dr(e),o=await Ei(t,this.loadOptions);return[s,o]}}qo.URL_SCHEME_REGEX=/^https?:\/\//;function Um(n){const e=n.lastIndexOf("/"),t=n.lastIndexOf("?"),s=n.substring(0,e),o=t>e?n.substring(t):"";return[s+"/",o]}function Br(n){return n.match(qo.URL_SCHEME_REGEX)!=null}const ki=(n,e)=>{if(typeof fetch>"u"&&(e==null||e.fetchFunc==null))return null;{let t=!0;if(Array.isArray(n)?t=n.every(s=>Br(s)):t=Br(n),t)return Ni(n,e)}return null};Y.registerSaveRouter(ki);Y.registerLoadRouter(ki);function Ni(n,e){return new qo(n,e)}function IR(n,e){return Ni(n,e)}function jo(n,e){const t=n.shape.length,s=e.shape.length;if(t<1)throw new Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${t}.`);if(s<1)throw new Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${s}.`);if(e.dtype!=="int32")throw new Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.shape[s-1]>t)throw new Error(`index innermost dimension length must be <= tensor rank; saw: ${e.shape[s-1]} vs. ${t}`);if(A(n.shape)===0)throw new Error(`Requested more than 0 entries, but input is empty. Input shape: ${n.shape}.`);const o=e.shape,r=o[o.length-1];let a=1;for(let d=0;d<o.length-1;++d)a*=o[d];const i=n.shape,c=o.slice();c.pop();let l=1;for(let d=r;d<t;++d)l*=i[d],c.push(i[d]);const u=[...J(n.shape).map(d=>d/l),1].slice(0,r);return[c,a,l,u]}const TR=Object.freeze(Object.defineProperty({__proto__:null,prepareAndValidate:jo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const no=-2,Wm=-1;function Ko(n,e,t){const s=n.shape.length;R(s===e.length,()=>`Error in slice${s}D: Length of begin ${e} must match the rank of the array (${s}).`),R(s===t.length,()=>`Error in slice${s}D: Length of size ${t} must match the rank of the array (${s}).`);for(let o=0;o<s;++o)R(e[o]+t[o]<=n.shape[o],()=>`Error in slice${s}D: begin[${o}] + size[${o}] (${e[o]+t[o]}) would overflow input.shape[${o}] (${n.shape[o]})`)}function Gm(n){const e=[];let t=0;for(;n>0;)n&1&&e.push(t),n/=2,t++;return e}function Ai(n,e,t){const s=[];for(let o=0;o<n.length;o++)s[o]=Math.ceil((e[o]-n[o])/t[o]);return s}function Fi(n,e,t,s){const o=[...n];for(let r=o.length;r<s.length;r++)o.push(1);for(let r=0;r<t;r++)r===0?o[e]=1:(o.splice(e,0,1),o.pop());return o}function Di(n,e,t){return t<=n?t:t-(e-1)}function Oi(n,e){const t=[];for(let s=0;s<n;s++)t.push(e+s);return t}function zm(n,e,t,s,o,r,a,i,c){const l=n.length;let u=new Array(l),d=new Array(l),h=new Array(l);if(e.length&&t>0){const f=e[0],p=t+1;u=Pi(a,f,p,s,n),d=_i(i,f,p,o,n),h=Fi(r,f,p,n)}else for(let f=0;f<l;f++)u[f]=Bi(a,s,r,n,f,c),d[f]=Mi(i,o,r,n,f,c),h[f]=Li(r,f,c);return{begin:u,end:d,strides:h}}function Pi(n,e,t,s,o){const r=[...o],a=Oi(t,e);for(let i=0;i<r.length;i++)if(a.indexOf(i)>-1)r[i]=0;else{const c=Di(e,t,i);let l=s[c];n&1<<c&&(l=0),r[i]=l}return r}function _i(n,e,t,s,o){const r=[...o],a=Oi(t,e);for(let i=0;i<r.length;i++)if(a.indexOf(i)>-1)r[i]=Number.MAX_SAFE_INTEGER;else{const c=Di(e,t,i);let l=s[c];n&1<<c&&(l=Number.MAX_SAFE_INTEGER),r[i]=l}for(let i=0;i<r.length;i++){const c=o[i];r[i]<0&&(r[i]+=c),r[i]=wn(0,r[i],o[i])}return r}function Li(n,e,t){let s=n[e];return(t&1<<e||s==null)&&(s=1),s}function Bi(n,e,t,s,o,r){let a=e[o];const i=t[o]||1;(n&1<<o||r&1<<o||a==null)&&(i>0?a=Number.MIN_SAFE_INTEGER:a=Number.MAX_SAFE_INTEGER);const c=s[o];return a<0&&(a+=c),a=wn(0,a,c-1),a}function Mi(n,e,t,s,o,r){let a=e[o];const i=t[o]||1;(n&1<<o||r&1<<o||a==null)&&(i>0?a=Number.MAX_SAFE_INTEGER:a=Number.MIN_SAFE_INTEGER);const c=s[o];return a<0&&(a+=c),i>0?a=wn(0,a,c):a=wn(-1,a,c-1),a}function Yo(n,e,t){let s=t.length;for(let o=0;o<t.length;o++)if(t[o]>1){s=o;break}for(let o=s+1;o<t.length;o++)if(e[o]>0||t[o]!==n[o])return!1;return!0}function Qo(n,e){let t=n.length>0?n[n.length-1]:1;for(let s=0;s<n.length-1;s++)t+=n[s]*e[s];return t}function Zo(n,e,t){let s;const o=n.shape.length;typeof e=="number"?s=[e,...new Array(o-1).fill(0)]:e.length<o?s=e.concat(new Array(o-e.length).fill(0)):s=e.slice(),s.forEach(a=>{R(a!==-1,()=>"slice() does not support negative begin indexing.")});let r;return t==null?r=new Array(o).fill(-1):typeof t=="number"?r=[t,...new Array(o-1).fill(-1)]:t.length<o?r=t.concat(new Array(o-t.length).fill(-1)):r=t,r=r.map((a,i)=>a>=0?a:(R(a===-1,()=>`Negative size values should be exactly -1 but got ${a} for the slice() size at index ${i}.`),n.shape[i]-s[i])),[s,r]}function Vi(n,e,t,s,o,r,a,i,c){let l;if(s==null?(l=new Array(e.length),l.fill(1)):l=s,a!=null&&(a&a-1)!==0)throw new Error("Multiple ellipses in slice is not allowed.");let u=!1;const d={dims:l.length,numAddAxisAfterEllipsis:0,begin:e.slice(),end:t.slice(),strides:l.slice(),beginMask:o,endMask:r,ellipsisMask:a,newAxisMask:i,shrinkAxisMask:c};for(let C=0;C<d.dims;C++)u&&(1<<C&i)!==0&&d.numAddAxisAfterEllipsis++,1<<C&a&&(u=!0);u||(d.ellipsisMask|=1<<d.dims,d.dims++);const h={dims:n.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};Hm(d,h);let f=!0,p=!0,x=!0;const g=[],m=[];for(let C=0;C<n.length;++C){if(h.strides[C]===0)throw Error(`strides[${C}] must be non-zero`);const y=!!(h.shrinkAxisMask&1<<C),S=n[C];if(S===-1){g.push(y?1:-1);continue}const N=[h.beginMask&1<<C,h.endMask&1<<C],k=[h.strides[C]>0?0:-1,h.strides[C]>0?S:S-1];if(y&&h.strides[C]<=0)throw Error("only stride 1 allowed on non-range indexing.");x=x&&h.strides[C]===1;const T=!!(h.beginMask&1<<C&&h.endMask&1<<C);if(h.beginValid&&h.endValid){if(y){const E=h.begin[C]<0?S+h.begin[C]:h.begin[C];if(h.begin[C]=E,h.end[C]=h.begin[C]+1,E<0||E>=S)throw Error(`slice index ${h.begin[C]} of dimension ${C} out of bounds.`)}else h.begin[C]=Mr(h.begin[C],0,h.strides[C],S,N,k),h.end[C]=Mr(h.end[C],1,h.strides[C],S,N,k);const w=h.strides[C]===1&&h.begin[C]===0&&h.end[C]===S;f=f&&w,p=p&&(C===0&&h.strides[C]===1||w)}else f=f&&h.strides[C]===1&&T,p=p&&(C===0&&h.strides[C]===1||T);let I,v=!1;if(h.beginValid&&h.endValid?(I=h.end[C]-h.begin[C],v=!0):y?(I=1,v=!0):T&&S>=0&&(h.strides[C]<0?I=-S:I=S,v=!0),v){let w;I===0||I<0!=h.strides[C]<0?w=0:w=Math.trunc(I/h.strides[C])+(I%h.strides[C]!==0?1:0),g.push(w)}else g.push(-1)}for(let C=0;C<h.finalShapeGatherIndices.length;++C){const y=h.finalShapeGatherIndices[C];y>=0?m.push(g[y]):y===no&&m.push(1)}return{finalShapeSparse:m.filter((C,y)=>h.finalShapeGatherIndices[y]!==no),finalShape:m,isIdentity:f,sliceDim0:p,isSimpleSlice:x,begin:h.begin,end:h.end,strides:h.strides}}function Hm(n,e){e.beginMask=0,e.endMask=0,e.shrinkAxisMask=0;let t=0;e.beginValid=n.begin!=null,e.endValid=n.end!=null,e.begin=new Array(e.dims),e.end=new Array(e.dims),e.strides=new Array(e.dims),e.finalShapeGatherIndices=[],e.finalShapeGatherIndicesSparse=[],e.inputShapeGatherIndicesSparse=new Array(e.dims);for(let s=0;s<n.dims;s++)if(1<<s&n.ellipsisMask){const o=Math.min(e.dims-(n.dims-s)+1+n.numAddAxisAfterEllipsis,e.dims);for(;t<o;t++)e.begin[t]=0,e.end[t]=0,e.strides[t]=1,e.beginMask|=1<<t,e.endMask|=1<<t,e.finalShapeGatherIndices.push(t),e.finalShapeGatherIndicesSparse.push(-1),e.inputShapeGatherIndicesSparse[t]=s}else if(1<<s&n.newAxisMask)e.finalShapeGatherIndices.push(no),e.finalShapeGatherIndicesSparse.push(-1);else{if(t===e.begin.length)throw Error(`Index out of range using input dim ${t}; input has only ${e.dims} dims, ${e.begin.length}.`);n.begin!=null&&(e.begin[t]=n.begin[s]),n.end!=null&&(e.end[t]=n.end[s]),e.strides[t]=n.strides[s],n.beginMask&1<<s&&(e.beginMask|=1<<t),n.endMask&1<<s&&(e.endMask|=1<<t),n.shrinkAxisMask&1<<s?(e.finalShapeGatherIndices.push(Wm),e.finalShapeGatherIndicesSparse.push(-1),e.shrinkAxisMask|=1<<t):(e.finalShapeGatherIndices.push(t),e.finalShapeGatherIndicesSparse.push(s)),e.inputShapeGatherIndicesSparse[t]=s,t++}}function Mr(n,e,t,s,o,r){if(o[e])return t>0?r[e]:r[e+1&1];{const a=n<0?s+n:n;return a<r[0]?r[0]:a>r[1]?r[1]:a}}const Xm=Object.freeze(Object.defineProperty({__proto__:null,assertParamsValid:Ko,computeFlatOffset:Qo,computeOutShape:Ai,getNormalizedAxes:zm,isSliceContinous:Yo,maskToAxes:Gm,parseSliceParams:Zo,sliceInfo:Vi,startForAxis:Bi,startIndicesWithElidedDims:Pi,stopForAxis:Mi,stopIndicesWithElidedDims:_i,stridesForAxis:Li,stridesWithElidedDims:Fi},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qm=typeof requestAnimationFrame<"u"?requestAnimationFrame:typeof setImmediate<"u"?setImmediate:n=>n();function jm(){return new Promise(n=>qm(()=>n()))}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ui(n,e){const t=n[0].length;n.forEach((o,r)=>{R(o.length===t,()=>`Error in concat${t}D: rank of tensors[${r}] must be the same as the rank of the rest (${t})`)}),R(e>=0&&e<t,()=>`Error in concat${t}D: axis must be between 0 and ${t-1}.`);const s=n[0];n.forEach((o,r)=>{for(let a=0;a<t;a++)R(a===e||o[a]===s[a],()=>`Error in concat${t}D: Shape of tensors[${r}] (${o}) does not match the shape of the rest (${s}) along the non-concatenated axis ${r}.`)})}function Tt(n,e){const t=n[0].slice();for(let s=1;s<n.length;s++)t[e]+=n[s][e];return t}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Oe;(function(n){n[n.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",n[n.VALUE_ROWIDS=1]="VALUE_ROWIDS",n[n.ROW_LENGTHS=2]="ROW_LENGTHS",n[n.ROW_SPLITS=3]="ROW_SPLITS",n[n.ROW_LIMITS=4]="ROW_LIMITS",n[n.ROW_STARTS=5]="ROW_STARTS"})(Oe||(Oe={}));function Wi(n,e,t){let s=new Array;if(t==null&&e==null)return s;if(e==null)for(;s.length<n+t.length;)s.push(-1);else s=e.slice();if(t==null)return s;if(n+t.length!==s.length)throw new Error(`rt input.shape and shape=${e} are incompatible: rt input.rank = ${n+t.length}, but shape.rank = ${s.length}`);for(let o=1;o<t.length;++o){const r=t[o],a=s[s.length-t.length+o],i=s[a];if(r>=0)if(i>=0){if(i!==r)throw new Error(`rt input.shape and shape=${e} are incompatible: rt input.shape[${o+n}] = ${r} but shape[${o+n}] = ${i}`)}else s[a]=r}return s}function Gi(n){const e={FIRST_DIM_SIZE:Oe.FIRST_DIM_SIZE,VALUE_ROWIDS:Oe.VALUE_ROWIDS,ROW_LENGTHS:Oe.ROW_LENGTHS,ROW_SPLITS:Oe.ROW_SPLITS,ROW_LIMITS:Oe.ROW_LIMITS,ROW_STARTS:Oe.ROW_STARTS},t=[];for(const s of n)if(s in e)t.push(e[s]);else break;return t}function zi(n){return n.length===0?0:n[0]===Oe.FIRST_DIM_SIZE?n.length-1:n.length}function Hi(n,e){if(n==null||e==null)return;const t=n.length,s=e.length;if(t>=s)throw new Error(`defaultValue.shape=${n} and ragged tensor flatValues.shape=${e}, are incompatible: defaultValue.rank = ${t} must be less than ragged tensor input flatValues.rank = ${s})`);for(let o=0;o<Math.min(t,s-1);++o){const r=n[o],a=e[o+1];if(r>=0&&a>=0&&r!==1&&r!==a)throw new Error(`defaultValue.shape=${n}, and ragged tensor input flatValues.shape=${e} are incompatible: defaultValue.shape[${o-n.length}] = ${r} but ragged tensor input.flatValues.shape[${o-n.length}] = ${a}`)}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jo=30;function Ts(n){return n<=Jo?n:as(n,Math.floor(Math.sqrt(n)))}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xi(n,e,t){const s=t*(typeof n=="number"?n:n[0]),o=e*(typeof n=="number"?n:n[1]);return[s,o]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function er(n,e,t,s=!0){let o=[];if(s)o=o.concat(e.slice(0)),o.push(n[0]/t),o=o.concat(n.slice(1));else{o=o.concat(n[0]);const r=e.length;for(let a=0;a<r;++a)o=o.concat([n[a+1]/e[a],e[a]]);o=o.concat(n.slice(r+1))}return o}function tr(n,e,t=!0){const s=[];if(t){s.push(e);for(let o=e+1;o<n;++o)o<=2*e?(s.push(o),s.push(o-(e+1))):s.push(o)}else{const o=[],r=[];for(let a=1;a<n;++a)a>=e*2+1||a%2===1?r.push(a):o.push(a);s.push(...o),s.push(0),s.push(...r)}return s}function nr(n,e,t,s=!0){const o=[];s?o.push(n[0]/t):o.push(n[0]*t);for(let r=1;r<n.length;++r)r<=e.length?s?o.push(e[r-1]*n[r]):o.push(n[r]/e[r-1]):o.push(n[r]);return o}function qi(n,e){const t=[0];for(let s=0;s<e;++s)t.push(n[s][0]);return t}function ji(n,e,t){const s=n.slice(0,1);for(let o=0;o<t;++o)s.push(n[o+1]-e[o][0]-e[o][1]);return s}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ki=1.7580993408473768,Yi=1.0507009873554805;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qi=.3275911,Zi=.254829592,Ji=-.284496736,ec=1.421413741,tc=-1.453152027,nc=1.061405429;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function In(n,e){if(n.length!==e.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${n.length}, imag: ${e.length}.`);const t=new Float32Array(n.length*2);for(let s=0;s<t.length;s+=2)t[s]=n[s/2],t[s+1]=e[s/2];return t}function Km(n){const e=new Float32Array(n.length/2),t=new Float32Array(n.length/2);for(let s=0;s<n.length;s+=2)e[s/2]=n[s],t[s/2]=n[s+1];return{real:e,imag:t}}function Ym(n){const e=Math.ceil(n.length/4),t=new Float32Array(e),s=new Float32Array(e);for(let o=0;o<n.length;o+=4)t[Math.floor(o/4)]=n[o],s[Math.floor(o/4)]=n[o+1];return{real:t,imag:s}}function Qm(n){const e=Math.floor(n.length/4),t=new Float32Array(e),s=new Float32Array(e);for(let o=2;o<n.length;o+=4)t[Math.floor(o/4)]=n[o],s[Math.floor(o/4)]=n[o+1];return{real:t,imag:s}}function Zm(n,e){const t=n[e*2],s=n[e*2+1];return{real:t,imag:s}}function Jm(n,e,t,s){n[s*2]=e,n[s*2+1]=t}function eg(n,e){const t=new Float32Array(n/2),s=new Float32Array(n/2);for(let o=0;o<Math.ceil(n/2);o++){const r=(e?2:-2)*Math.PI*(o/n);t[o]=Math.cos(r),s[o]=Math.sin(r)}return{real:t,imag:s}}function tg(n,e,t){const s=(t?2:-2)*Math.PI*(n/e),o=Math.cos(s),r=Math.sin(s);return{real:o,imag:r}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ms="->",ng=/->/g,Vr=",",Ur="...";function sc(n,e){n=n.replace(/\s/g,"");const t=(n.length-n.replace(ng,"").length)/Ms.length;if(t<1)throw new Error("Equations without an arrow are not supported.");if(t>1)throw new Error(`Equation must contain exactly one arrow ("${Ms}").`);const[s,o]=n.split(Ms);R(s.indexOf(Ur)===-1,()=>`The ellipsis notation ("${Ur}") is not supported yet.`);const r=s.split(Vr),a=r.length;if(e!==a)throw new Error(`Expected ${a} input tensors, received ${e}`);if(a>2)throw new Error("Support for more than 2 input tensors is not implemented yet.");const i=[];for(let h=0;h<o.length;++h){const f=o[h];if(!r.some(p=>p.indexOf(f)!==-1))throw new Error(`Output subscripts contain the label ${f} not present in the input subscripts.`);i.indexOf(f)===-1&&i.push(f)}for(let h=0;h<s.length;++h){const f=s[h];i.indexOf(f)===-1&&f!==Vr&&i.push(f)}const c=new Array(r.length);for(let h=0;h<a;++h){if(new Set(r[h].split("")).size!==r[h].length)throw new Error(`Found duplicate axes in input component ${r[h]}. Support for duplicate axes in input is not implemented yet.`);c[h]=[];for(let f=0;f<r[h].length;++f)c[h].push(i.indexOf(r[h][f]))}const l=i.length,u=o.length,d=[];for(let h=u;h<l;++h)d.push(h);return{allDims:i,summedDims:d,idDims:c}}function oc(n,e){let t=new Array(n);t.fill(-1);for(let o=0;o<e.length;++o)t[e[o]]=o;const s=[];for(let o=0;o<n;++o)t[o]===-1&&s.push(o);return t=t.filter(o=>o!==-1),{permutationIndices:t,expandDims:s}}function rc(n,e,t){const s=new Array(n);for(let o=0;o<t.length;++o){const r=t[o].shape;for(let a=0;a<e[o].length;++a)s[e[o][a]]===void 0?s[e[o][a]]=r[a]:R(s[e[o][a]]===r[a],()=>`Expected dimension ${s[e[o][a]]} at axis ${a} of input shaped ${JSON.stringify(r)}, but got dimension ${r[a]}`)}}function ac(n,e){const t=n,s=[];let o=0;n.length===0&&t.push(-1),o=n.length+1;for(let a=0;a<o;++a)s.push([]);const r=[];for(let a=0;a<t.length;++a){const i=t[a],c=sg(e,i);for(const l of c)r.indexOf(l)===-1&&(s[a].push(l),r.push(l))}return{path:t,steps:s}}function ic(n){return n.every((e,t)=>e===t)}function sg(n,e){const t=[];for(let s=0;s<n.length;++s)(n[s].length===0||n[s].indexOf(e)!==-1||e===-1)&&t.push(s);return t}function cc(n,e,t=0){let s=[];if(typeof e=="number")R(n.shape[t]%e===0,()=>"Number of splits must evenly divide the axis."),s=new Array(e).fill(n.shape[t]/e);else{const o=e.reduce((a,i)=>(i===-1&&(a+=1),a),0);R(o<=1,()=>"There should be only one negative value in split array.");const r=e.indexOf(-1);if(r!==-1){const a=e.reduce((i,c)=>c>0?i+c:i);e[r]=n.shape[t]-a}R(n.shape[t]===e.reduce((a,i)=>a+i),()=>"The sum of sizes must match the size of the axis dimension."),s=e}return s}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lc(n){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${n}`}function uc(n,e){return`indices(${n}, 0) is invalid: ${e} < 0`}function dc(n,e,t){return`indices(${n}, 0) is invalid: ${e} >= ${t}`}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hc(n,e){return`only one output dimension may be -1, not both ${n} and ${e}`}function fc(n,e){return`size ${n} must be non-negative, not ${e}`}function pc(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function mc(n,e){const t=A(n),s=A(e);return`Input to reshape is a SparseTensor with ${t}
  dense values, but the requested shape requires a multiple of ${s}. inputShape=${n} outputShape= ${e}`}function gc(n,e){const t=A(n),s=A(e);return`Input to reshape is a tensor with ${t} dense values, but the requested shape has ${s}. inputShape=${n} outputShape=${e}`}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function so(){return"segment ids must be >= 0"}function xc(){return"segment ids are not increasing"}function bc(n,e){return`Segment id ${n} out of range [0, ${e}), possibly because segmentIds input is not sorted.`}function wc(n,e,t){return`Bad: indices[${n}] == ${e} out of range [0, ${t})`}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cc(n,e){let t=!1,s;for(n<=Jo?(s=n,t=!0):s=as(n,Math.floor(Math.sqrt(n)));!t;)s>e||s===n?t=!0:s=as(n,s+1);return s}function yc(n,e,t){const s=[],o=n.length;for(let r=0;r<o;r++)r!==e?s.push(n[r]):s.push(t);return s}function $c(n,e,t,s){const o=e.shape.length,r=n.shape.length;if(s!==0&&(s<-o||s>o))throw new Error(`Expect batchDims in the range of [-${o}, ${o}], but got ${s}`);if(s<0&&(s+=o),s>r)throw new Error(`batchDims (${s}) must be less than rank(x) (
    ${r}).`);if(t<s)throw new Error(`batchDims (${s}) must be less than or equal to axis (${t}).`);for(let d=0;d<s;++d)if(n.shape[d]!==e.shape[d])throw new Error(`x.shape[${d}]: ${n.shape[d]} should be equal to indices.shape[${d}]: ${e.shape[d]}.`);const a=n.shape[t],i=[];let c=1,l=1,u=1;for(let d=0;d<s;++d)i.push(n.shape[d]),c*=n.shape[d];for(let d=s;d<t;d++)i.push(n.shape[d]),l*=n.shape[d];for(let d=s;d<o;d++)i.push(e.shape[d]);for(let d=t+1;d<r;d++)i.push(n.shape[d]),u*=n.shape[d];return{batchSize:c,sliceSize:u,outerSize:l,dimSize:a,outputShape:i}}const og=Object.freeze(Object.defineProperty({__proto__:null,collectGatherOpShapeInfo:$c,computeOutShape:yc,segOpComputeOptimalWindowSize:Cc},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ye(n){try{return n.map(e=>Ct(e))}catch(e){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${e}`)}}function vc(n){return n.map(e=>ot(e))}const rg=Object.freeze(Object.defineProperty({__proto__:null,ERF_A1:Zi,ERF_A2:Ji,ERF_A3:ec,ERF_A4:tc,ERF_A5:nc,ERF_P:Qi,PARALLELIZE_THRESHOLD:Jo,get RowPartitionType(){return Oe},SELU_SCALE:Yi,SELU_SCALEALPHA:Ki,applyActivation:um,assertAndGetBroadcastShape:ae,assertAxesAreInnerMostDims:Le,assertParamsConsistent:Ui,assignToTypedArray:Jm,axesAreInnerMostDims:Go,calculateShapes:Mn,checkEinsumDimSizes:rc,checkPadOnDimRoundingMode:Fp,combineLocations:vi,combineRaggedTensorToTensorShapes:Wi,complexWithEvenIndex:Ym,complexWithOddIndex:Qm,computeConv2DInfo:_e,computeConv3DInfo:Bn,computeDefaultPad:Vo,computeDilation2DInfo:$i,computeOptimalWindowSize:Ts,computeOutAndReduceShapes:He,computeOutShape:Tt,computePool2DInfo:Zt,computePool3DInfo:Ln,convertConv2DDataFormat:en,decodeEinsumEquation:sc,eitherStridesOrDilationsAreOne:Jt,expandShapeToKeepDim:Xe,exponent:tg,exponents:eg,fromStringArrayToUint8:vc,fromUint8ToStringArray:Ye,getAxesPermutation:Re,getBroadcastDims:St,getComplexWithIndex:Zm,getEinsumComputePath:ac,getEinsumPermutation:oc,getFusedBiasGradient:lm,getFusedDyActivation:cm,getImageCenter:Xi,getInnerMostAxes:Ee,getPermuted:tr,getRaggedRank:zi,getReductionAxes:Wo,getReshaped:er,getReshapedPermuted:nr,getRowPartitionTypesHelper:Gi,getSliceBeginCoords:qi,getSliceSize:ji,getSparseFillEmptyRowsIndicesDenseShapeMismatch:lc,getSparseFillEmptyRowsNegativeIndexErrorMessage:uc,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:dc,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:pc,getSparseReshapeInputOutputMismatchErrorMessage:gc,getSparseReshapeInputOutputMultipleErrorMessage:mc,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:hc,getSparseReshapeNegativeOutputDimErrorMessage:fc,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:wc,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:so,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:xc,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:bc,getUndoAxesPermutation:zo,isIdentityPermutation:ic,log:df,mergeRealAndImagArrays:In,prepareAndValidate:jo,prepareSplitSize:cc,segment_util:og,shouldFuse:dm,slice_util:Xm,splitRealAndImagArrays:Km,stridesOrDilationsArePositive:Ap,tupleValuesAreOne:Js,upcastType:Ge,validateDefaultValueShape:Hi,validateInput:Si,validateUpdateShape:Ho,warn:Ae},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Nm();/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gt={},Xn={alpha:!1,antialias:!1,premultipliedAlpha:!1,preserveDrawingBuffer:!1,depth:!1,stencil:!1,failIfMajorPerformanceCaveat:!0};function Sc(n,e){gt[n]=e}function Pe(n,e){if(!(n in gt)||e!=null){const s=ig(n,e);if(s!==null)gt[n]=s;else return console.log("Could not get context for WebGL version",n),null}const t=gt[n];return t==null||t.isContextLost()?(delete gt[n],Pe(n)):(t.disable(t.DEPTH_TEST),t.disable(t.STENCIL_TEST),t.disable(t.BLEND),t.disable(t.DITHER),t.disable(t.POLYGON_OFFSET_FILL),t.disable(t.SAMPLE_COVERAGE),t.enable(t.SCISSOR_TEST),t.enable(t.CULL_FACE),t.cullFace(t.BACK),gt[n])}function ag(n){if(!$().getBool("IS_SAFARI")&&typeof OffscreenCanvas<"u"&&n===2)return new OffscreenCanvas(300,150);if(typeof document<"u")return document.createElement("canvas");throw new Error("Cannot create a canvas in this context")}function ig(n,e){if(n!==1&&n!==2)throw new Error("Cannot get WebGL rendering context, WebGL is disabled.");const t=e??ag(n);return t.addEventListener("webglcontextlost",s=>{s.preventDefault(),delete gt[n]},!1),$().getBool("SOFTWARE_WEBGL_ENABLED")&&(Xn.failIfMajorPerformanceCaveat=!1),n===1?t.getContext("webgl",Xn)||t.getContext("experimental-webgl",Xn):t.getContext("webgl2",Xn)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Tn;(function(n){n[n.DENSE=0]="DENSE",n[n.SHARED_BATCH=1]="SHARED_BATCH"})(Tn||(Tn={}));var ve;(function(n){n[n.RENDER=0]="RENDER",n[n.UPLOAD=1]="UPLOAD",n[n.PIXELS=2]="PIXELS",n[n.DOWNLOAD=3]="DOWNLOAD"})(ve||(ve={}));var re;(function(n){n[n.UNPACKED_FLOAT16=0]="UNPACKED_FLOAT16",n[n.UNPACKED_FLOAT32=1]="UNPACKED_FLOAT32",n[n.PACKED_4X1_UNSIGNED_BYTE=2]="PACKED_4X1_UNSIGNED_BYTE",n[n.PACKED_2X2_FLOAT32=3]="PACKED_2X2_FLOAT32",n[n.PACKED_2X2_FLOAT16=4]="PACKED_2X2_FLOAT16"})(re||(re={}));function Vn(n,e){return[e,n]}function cg(n,e){return n*e}function qn(n){const e=A(n),t=Math.ceil(e/4);return rs(t)}function tn(n,e){return[Math.max(1,Math.ceil(e/2)),Math.max(1,Math.ceil(n/2))]}function lg(n,e){const[t,s]=tn(n,e);return t*s*4}function sr(n,e){const t=n;let s,o,r,a,i,c,l,u,d,h;return $().getNumber("WEBGL_VERSION")===2?(s=t.R32F,o=t.R16F,r=t.RGBA16F,a=t.RGBA32F,i=t.RED,l=4,u=1,d=t.HALF_FLOAT,h=t.FLOAT,c=t.RGBA8):(s=n.RGBA,o=n.RGBA,r=n.RGBA,a=t.RGBA,i=n.RGBA,l=4,u=4,d=e!=null?e.HALF_FLOAT_OES:null,h=n.FLOAT,c=n.RGBA),{internalFormatFloat:s,internalFormatHalfFloat:o,internalFormatPackedHalfFloat:r,internalFormatPackedFloat:a,textureFormatFloat:i,downloadTextureFormat:c,downloadUnpackNumChannels:l,defaultNumChannels:u,textureTypeHalfFloat:d,textureTypeFloat:h}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function D(n,e){const t=e();return $().getBool("DEBUG")&&ug(n),t}function ug(n){const e=n.getError();if(e!==n.NO_ERROR)throw new Error("WebGL Error: "+Tc(n,e))}const dg=596e-10,hg=65504;function Ic(n){return!!($().getBool("WEBGL_RENDER_FLOAT32_ENABLED")||n===0||dg<Math.abs(n)&&Math.abs(n)<hg)}function Tc(n,e){switch(e){case n.NO_ERROR:return"NO_ERROR";case n.INVALID_ENUM:return"INVALID_ENUM";case n.INVALID_VALUE:return"INVALID_VALUE";case n.INVALID_OPERATION:return"INVALID_OPERATION";case n.INVALID_FRAMEBUFFER_OPERATION:return"INVALID_FRAMEBUFFER_OPERATION";case n.OUT_OF_MEMORY:return"OUT_OF_MEMORY";case n.CONTEXT_LOST_WEBGL:return"CONTEXT_LOST_WEBGL";default:return`Unknown error code ${e}`}}function pn(n,e){return Qe(n,()=>n.getExtension(e),'Extension "'+e+'" not supported on this browser.')}function Rc(n,e){const t=Qe(n,()=>n.createShader(n.VERTEX_SHADER),"Unable to create vertex WebGLShader.");if(D(n,()=>n.shaderSource(t,e)),D(n,()=>n.compileShader(t)),n.getShaderParameter(t,n.COMPILE_STATUS)===!1)throw console.log(n.getShaderInfoLog(t)),new Error("Failed to compile vertex shader.");return t}function Ec(n,e){const t=Qe(n,()=>n.createShader(n.FRAGMENT_SHADER),"Unable to create fragment WebGLShader.");if(D(n,()=>n.shaderSource(t,e)),D(n,()=>n.compileShader(t)),$().get("ENGINE_COMPILE_ONLY"))return t;if(n.getShaderParameter(t,n.COMPILE_STATUS)===!1)throw or(e,n.getShaderInfoLog(t)),new Error("Failed to compile fragment shader.");return t}const fg=/ERROR: [0-9]+:([0-9]+):/g;function or(n,e){const t=fg.exec(e);if(t==null){console.log(`Couldn't parse line number in error: ${e}`),console.log(n);return}const s=+t[1],o=n.split(`
`),r=o.length.toString().length+2,a=o.map((d,h)=>bt((h+1).toString(),r)+d);let i=0;for(let d=0;d<a.length;d++)i=Math.max(a[d].length,i);const c=a.slice(0,s-1),l=a.slice(s-1,s),u=a.slice(s);console.log(c.join(`
`)),console.log(e.split(`
`)[0]),console.log(`%c ${bt(l[0],i)}`,"border:1px solid red; background-color:#e3d2d2; color:#a61717"),console.log(u.join(`
`))}function kc(n){return Qe(n,()=>n.createProgram(),"Unable to create WebGLProgram.")}function Nc(n,e){if(D(n,()=>n.linkProgram(e)),!$().get("ENGINE_COMPILE_ONLY")&&n.getProgramParameter(e,n.LINK_STATUS)===!1)throw console.log(n.getProgramInfoLog(e)),new Error("Failed to link vertex and fragment shaders.")}function Jn(n,e){if(D(n,()=>n.validateProgram(e)),n.getProgramParameter(e,n.VALIDATE_STATUS)===!1)throw console.log(n.getProgramInfoLog(e)),new Error("Shader program validation failed.")}function Ac(n,e){const t=Qe(n,()=>n.createBuffer(),"Unable to create WebGLBuffer");return D(n,()=>n.bindBuffer(n.ARRAY_BUFFER,t)),D(n,()=>n.bufferData(n.ARRAY_BUFFER,e,n.STATIC_DRAW)),t}function Fc(n,e){const t=Qe(n,()=>n.createBuffer(),"Unable to create WebGLBuffer");return D(n,()=>n.bindBuffer(n.ELEMENT_ARRAY_BUFFER,t)),D(n,()=>n.bufferData(n.ELEMENT_ARRAY_BUFFER,e,n.STATIC_DRAW)),t}function pg(){return $().getNumber("WEBGL_VERSION")===2?1:4}function Dc(n){return Qe(n,()=>n.createTexture(),"Unable to create WebGLTexture.")}function Oc(n,e){const t=$().getNumber("WEBGL_MAX_TEXTURE_SIZE");if(n<=0||e<=0){const s=`[${n}x${e}]`;throw new Error("Requested texture size "+s+" is invalid.")}if(n>t||e>t){const s=`[${n}x${e}]`,o=`[${t}x${t}]`;throw new Error("Requested texture size "+s+" greater than WebGL maximum on this browser / GPU "+o+".")}}function Pc(n){return Qe(n,()=>n.createFramebuffer(),"Unable to create WebGLFramebuffer.")}function oo(n,e,t,s,o,r,a){const i=n.getAttribLocation(e,t);return i===-1?!1:(D(n,()=>n.bindBuffer(n.ARRAY_BUFFER,s)),D(n,()=>n.vertexAttribPointer(i,o,n.FLOAT,!1,r,a)),D(n,()=>n.enableVertexAttribArray(i)),!0)}function _c(n,e,t){Uc(n,t),D(n,()=>n.activeTexture(n.TEXTURE0+t)),D(n,()=>n.bindTexture(n.TEXTURE_2D,e))}function mg(n,e){Uc(n,e),D(n,()=>n.activeTexture(n.TEXTURE0+e)),D(n,()=>n.bindTexture(n.TEXTURE_2D,null))}function Lc(n,e,t){return Qe(n,()=>n.getUniformLocation(e,t),'uniform "'+t+'" not present in program.')}function Bc(n,e,t){return n.getUniformLocation(e,t)}function Mc(n,e,t,s){D(n,()=>_c(n,e,s)),D(n,()=>n.uniform1i(t,s))}function gg(n){D(n,()=>n.bindFramebuffer(n.FRAMEBUFFER,null)),D(n,()=>n.viewport(0,0,n.canvas.width,n.canvas.height)),D(n,()=>n.scissor(0,0,n.canvas.width,n.canvas.height))}function es(n,e,t){D(n,()=>n.bindFramebuffer(n.FRAMEBUFFER,t)),D(n,()=>n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,e,0))}function ro(n,e){D(n,()=>n.bindFramebuffer(n.FRAMEBUFFER,e)),D(n,()=>n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,null,0))}function mn(n){const e=n.checkFramebufferStatus(n.FRAMEBUFFER);if(e!==n.FRAMEBUFFER_COMPLETE)throw new Error("Error binding framebuffer: "+Vc(n,e))}function Vc(n,e){switch(e){case n.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_ATTACHMENT";case n.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";case n.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:return"FRAMEBUFFER_INCOMPLETE_DIMENSIONS";case n.FRAMEBUFFER_UNSUPPORTED:return"FRAMEBUFFER_UNSUPPORTED";default:return`unknown error ${e}`}}function Qe(n,e,t){const s=D(n,()=>e());if(s==null)throw new Error(t);return s}function Uc(n,e){const t=n.MAX_COMBINED_TEXTURE_IMAGE_UNITS-1,s=e+n.TEXTURE0;if(s<n.TEXTURE0||s>t){const o=`[gl.TEXTURE0, gl.TEXTURE${t}]`;throw new Error(`textureUnit must be in ${o}.`)}}function Rt(n,e=2){return A(n.slice(0,n.length-e))}function Et(n){if(n.length===0)throw Error("Cannot get rows and columns of an empty shape array.");return[n.length>1?n[n.length-2]:1,n[n.length-1]]}function gn(n){let e=[1,1,1];return n.length===0||n.length===1&&n[0]===1||(e=[Rt(n),...Et(n)]),e}function Wc(n,e=!1){let t=$().getNumber("WEBGL_MAX_TEXTURE_SIZE"),s=$().getNumber("WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE");s===1/0&&$().getBool("WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE")&&(s=t/2),e&&(t=t*2,s=s*2,n=n.map((i,c)=>c>=n.length-2?ps(n[c]):n[c]),n.length===1&&(n=[2,n[0]])),n.length!==2&&(n=lt(n).newShape);let o=A(n),r=null;n.length<=1&&o<=t?r=[1,o]:n.length===2&&n[0]<=t&&n[1]<=t?r=n:n.length===3&&n[0]*n[1]<=t&&n[2]<=t?r=[n[0]*n[1],n[2]]:n.length===3&&n[0]<=t&&n[1]*n[2]<=t?r=[n[0],n[1]*n[2]]:n.length===4&&n[0]*n[1]*n[2]<=t&&n[3]<=t?r=[n[0]*n[1]*n[2],n[3]]:n.length===4&&n[0]<=t&&n[1]*n[2]*n[3]<=t&&(r=[n[0],n[1]*n[2]*n[3]]);const a=r!=null&&Math.max(...r)>s&&Math.min(...r)<=(e?2:1)&&Math.min(...r)>0;if(r==null||a)if(e){const i=Rt(n);let c=2,l=2;n.length&&([c,l]=Et(n)),o=i*(c/2)*(l/2),r=rs(o).map(u=>u*2)}else r=rs(o);return r}function jn(n){return n%2===0}function Rn(n,e){if(n=n.slice(-2),e=e.slice(-2),oe(n,e)||!n.length||!e.length||n[0]===0||n[1]===0||e[0]===0||e[1]===0)return!0;if(n.length!==e.length){const t=n[n.length-1],s=e[e.length-1];if(t===s||jn(t)&&jn(s)&&(n[0]===1||e[0]===1))return!0}return n[1]===e[1]&&jn(n[0])&&jn(e[0])}let ts,ns;function Gc(n){if(ts==null){const e=Pe(n);ts=e.getParameter(e.MAX_TEXTURE_SIZE)}return ts}function xg(){ts=null}function bg(){ns=null}function zc(n){if(ns==null){const e=Pe(n);ns=e.getParameter(e.MAX_TEXTURE_IMAGE_UNITS)}return Math.min(16,ns)}function Hc(n){if(n===0)return 0;let e;const t=Pe(n);return Se(t,"EXT_disjoint_timer_query_webgl2")&&n===2?e=2:Se(t,"EXT_disjoint_timer_query")?e=1:e=0,e}function Se(n,e){return n.getExtension(e)!=null}function ao(n){try{if(Pe(n)!=null)return!0}catch(e){return console.log("Error when getting WebGL context: ",e),!1}return!1}function Xc(n){if(n===0)return!1;const e=Pe(n);if(n===1){if(!Se(e,"OES_texture_float"))return!1}else if(!Se(e,"EXT_color_buffer_float"))return!1;return io(e)}function qc(n){if(n===0)return!1;const e=Pe(n);if(n===1){if(!Se(e,"OES_texture_float")||!Se(e,"WEBGL_color_buffer_float"))return!1}else{if(Se(e,"EXT_color_buffer_float"))return io(e);const s="EXT_color_buffer_half_float";if(Se(e,s)){const o=e.getExtension(s);return wg(e,o)}return!1}return io(e)}function io(n){const e=sr(n),t=n.createTexture();n.bindTexture(n.TEXTURE_2D,t),n.texImage2D(n.TEXTURE_2D,0,e.internalFormatFloat,1,1,0,e.textureFormatFloat,e.textureTypeFloat,null);const r=n.createFramebuffer();n.bindFramebuffer(n.FRAMEBUFFER,r),n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,t,0);const a=n.checkFramebufferStatus(n.FRAMEBUFFER)===n.FRAMEBUFFER_COMPLETE;return n.bindTexture(n.TEXTURE_2D,null),n.bindFramebuffer(n.FRAMEBUFFER,null),n.deleteTexture(t),n.deleteFramebuffer(r),a}function wg(n,e){const t=sr(n,e),s=n.createTexture();n.bindTexture(n.TEXTURE_2D,s),n.texImage2D(n.TEXTURE_2D,0,t.internalFormatHalfFloat,1,1,0,t.textureFormatFloat,t.textureTypeHalfFloat,null);const a=n.createFramebuffer();n.bindFramebuffer(n.FRAMEBUFFER,a),n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,s,0);const i=n.checkFramebufferStatus(n.FRAMEBUFFER)===n.FRAMEBUFFER_COMPLETE;return n.bindTexture(n.TEXTURE_2D,null),n.bindFramebuffer(n.FRAMEBUFFER,null),n.deleteTexture(s),n.deleteFramebuffer(a),i}function jc(n){return n!==2?!1:Pe(n).fenceSync!=null}function nn(n,e){Array.isArray(n)||(n=[n]),n.forEach(t=>{t!=null&&R(t.dtype!=="complex64",()=>`${e} does not support complex64 tensors in the WebGL backend.`)})}const Cg=Object.freeze(Object.defineProperty({__proto__:null,assertNotComplex:nn,bindCanvasToFramebuffer:gg,bindColorTextureToFramebuffer:es,bindTextureToProgramUniformSampler:Mc,bindTextureUnit:_c,bindVertexBufferToProgramAttribute:oo,callAndCheck:D,canBeRepresented:Ic,createFragmentShader:Ec,createFramebuffer:Pc,createProgram:kc,createStaticIndexBuffer:Fc,createStaticVertexBuffer:Ac,createTexture:Dc,createVertexShader:Rc,getBatchDim:Rt,getExtensionOrThrow:pn,getFramebufferErrorMessage:Vc,getMaxTexturesInShader:zc,getNumChannels:pg,getProgramUniformLocation:Bc,getProgramUniformLocationOrThrow:Lc,getRowsCols:Et,getShapeAs3D:gn,getTextureShapeFromLogicalShape:Wc,getWebGLDisjointQueryTimerVersion:Hc,getWebGLErrorMessage:Tc,getWebGLMaxTextureSize:Gc,hasExtension:Se,isCapableOfRenderingToFloatTexture:Xc,isDownloadFloatTextureEnabled:qc,isReshapeFree:Rn,isWebGLFenceEnabled:jc,isWebGLVersionEnabled:ao,linkProgram:Nc,logShaderSourceAndInfoLog:or,resetMaxTextureSize:xg,resetMaxTexturesInShader:bg,unbindColorTextureFromFramebuffer:ro,unbindTextureUnit:mg,validateFramebuffer:mn,validateProgram:Jn,validateTextureSize:Oc},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const P=$();P.registerFlag("HAS_WEBGL",()=>P.getNumber("WEBGL_VERSION")>0);P.registerFlag("WEBGL_VERSION",()=>ao(2)?2:ao(1)?1:0);P.registerFlag("WEBGL_CHECK_NUMERICAL_PROBLEMS",()=>!1);P.registerFlag("WEBGL_BUFFER_SUPPORTED",()=>P.get("WEBGL_VERSION")===2);P.registerFlag("WEBGL_CPU_FORWARD",()=>!0);P.registerFlag("WEBGL_FORCE_F16_TEXTURES",()=>!1);P.registerFlag("WEBGL_PACK",()=>P.getBool("HAS_WEBGL"));P.registerFlag("WEBGL_PACK_NORMALIZATION",()=>P.getBool("WEBGL_PACK"));P.registerFlag("WEBGL_PACK_CLIP",()=>P.getBool("WEBGL_PACK"));P.registerFlag("WEBGL_PACK_DEPTHWISECONV",()=>P.getBool("WEBGL_PACK"));P.registerFlag("WEBGL_PACK_BINARY_OPERATIONS",()=>P.getBool("WEBGL_PACK"));P.registerFlag("WEBGL_PACK_UNARY_OPERATIONS",()=>P.getBool("WEBGL_PACK"));P.registerFlag("WEBGL_PACK_ARRAY_OPERATIONS",()=>P.getBool("WEBGL_PACK"));P.registerFlag("WEBGL_PACK_IMAGE_OPERATIONS",()=>P.getBool("WEBGL_PACK"));P.registerFlag("WEBGL_PACK_REDUCE",()=>P.getBool("WEBGL_PACK"));P.registerFlag("WEBGL_LAZILY_UNPACK",()=>P.getBool("WEBGL_PACK"));P.registerFlag("WEBGL_CONV_IM2COL",()=>P.getBool("WEBGL_PACK"));P.registerFlag("WEBGL_PACK_CONV2DTRANSPOSE",()=>P.getBool("WEBGL_PACK"));P.registerFlag("WEBGL_MAX_TEXTURE_SIZE",()=>Gc(P.getNumber("WEBGL_VERSION")));P.registerFlag("WEBGL_MAX_TEXTURES_IN_SHADER",()=>zc(P.getNumber("WEBGL_VERSION")));P.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION",()=>{const n=P.getNumber("WEBGL_VERSION");return n===0?0:Hc(n)});P.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE",()=>P.getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0&&!Lo());P.registerFlag("WEBGL_RENDER_FLOAT32_CAPABLE",()=>Xc(P.getNumber("WEBGL_VERSION")));P.registerFlag("WEBGL_RENDER_FLOAT32_ENABLED",()=>P.getBool("WEBGL_FORCE_F16_TEXTURES")?!1:P.getBool("WEBGL_RENDER_FLOAT32_CAPABLE"));P.registerFlag("WEBGL_DOWNLOAD_FLOAT_ENABLED",()=>qc(P.getNumber("WEBGL_VERSION")));P.registerFlag("WEBGL_FENCE_API_ENABLED",()=>jc(P.getNumber("WEBGL_VERSION")));P.registerFlag("WEBGL_SIZE_UPLOAD_UNIFORM",()=>P.getBool("WEBGL_RENDER_FLOAT32_ENABLED")?4:0);P.registerFlag("WEBGL_DELETE_TEXTURE_THRESHOLD",()=>-1,n=>{if(typeof n!="number")throw new Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be a number but got ${n}.`);if(n<0&&n!==-1)throw new Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be -1 (indicating never delete) or at least 0, but got ${n}.`)});P.registerFlag("WEBGL_FLUSH_THRESHOLD",()=>Lo()?1:-1,n=>{if(typeof n!="number")throw new Error(`WEBGL_FLUSH_THRESHOLD must be a number but got ${n}.`);if(n<0&&n!==-1)throw new Error(`WEBGL_FLUSH_THRESHOLD must be -1 (indicating never manual flush) or at least 0, but got ${n}.`)});P.registerFlag("CPU_HANDOFF_SIZE_THRESHOLD",()=>128);P.registerFlag("WEBGL_USE_SHAPES_UNIFORMS",()=>!1);P.registerFlag("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD",()=>1e5);P.registerFlag("TOPK_K_CPU_HANDOFF_THRESHOLD",()=>128);P.registerFlag("WEBGL_EXP_CONV",()=>!1);P.registerFlag("SOFTWARE_WEBGL_ENABLED",()=>P.getBool("IS_TEST"));P.registerFlag("WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE",()=>1/0);P.registerFlag("WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE",()=>!1);P.registerFlag("WEBGL2_ISNAN_CUSTOM",()=>!1);P.registerFlag("ENGINE_COMPILE_ONLY",()=>!1);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ge(){let n,e,t,s,o,r,a,i,c,l;return $().getNumber("WEBGL_VERSION")===2?(n="#version 300 es",e="in",t="out",s="in",o="texture",r="outputColor",a="out vec4 outputColor;",i=$().getBool("WEBGL2_ISNAN_CUSTOM")?`
      bool isnan_custom(float val) {
        uint floatToUint = floatBitsToUint(val);
        return (floatToUint & 0x7fffffffu) > 0x7f800000u;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `:"",c="",l=`
      #define round(value) newRound(value)
      int newRound(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 newRound(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `):(n="",e="attribute",t="varying",s="varying",o="texture2D",r="gl_FragColor",a="",i=`
      #define isnan(value) isnan_custom(value)
      bool isnan_custom(float val) {
        return (val > 0. || val < 1. || val == 0.) ? false : true;
      }
      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));
      }
    `,c=`
      uniform float INFINITY;

      bool isinf(float val) {
        return abs(val) == INFINITY;
      }
      bvec4 isinf(vec4 val) {
        return equal(abs(val), vec4(INFINITY));
      }
    `,l=`
      int round(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 round(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `),{version:n,attribute:e,varyingVs:t,varyingFs:s,texture2D:o,output:r,defineOutput:a,defineSpecialNaN:i,defineSpecialInf:c,defineRound:l}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ft(n,e,t="index"){const s=J(e);return s.map((o,r)=>{const a=`int ${n[r]} = ${t} / ${o}`,i=r===s.length-1?`int ${n[r+1]} = ${t} - ${n[r]} * ${o}`:`index -= ${n[r]} * ${o}`;return`${a}; ${i};`}).join("")}function Rs(n,e,t="index"){const s=J(e);return s.map((o,r)=>{const a=`int ${n[r]} = ${t} / outShapeStrides[${r}]`,i=r===s.length-1?`int ${n[r+1]} = ${t} - ${n[r]} * outShapeStrides[${r}]`:`index -= ${n[r]} * outShapeStrides[${r}]`;return`${a}; ${i};`}).join("")}function yg(n,e){const t=n.length,s=n.map(r=>`${e}[${r}]`),o=new Array(t-1);o[t-2]=s[t-1];for(let r=t-3;r>=0;--r)o[r]=`(${o[r+1]} * ${s[r+1]})`;return o}function $g(n,e,t="index"){const s=n.map((r,a)=>a),o=yg(s,e);return o.map((r,a)=>{const i=`int ${n[a]} = ${t} / ${o[a]}`,c=a===o.length-1?`int ${n[a+1]} = ${t} - ${n[a]} * ${o[a]}`:`index -= ${n[a]} * ${o[a]}`;return`${i}; ${c};`}).join("")}function rr(n){const e=J(n).map(t=>t.toString());return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * ${e[0]} + coords.y * ${e[1]} + coords.z;
  }
`}function ar(){return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * outShapeStrides[0] + coords.y * outShapeStrides[1] + coords.z;
  }
`}const Kc=`
  const float FLOAT_MAX = 1.70141184e38;
  const float FLOAT_MIN = 1.17549435e-38;

  lowp vec4 encode_float(highp float v) {
    if (isnan(v)) {
      return vec4(255, 255, 255, 255);
    }

    highp float av = abs(v);

    if(av < FLOAT_MIN) {
      return vec4(0.0, 0.0, 0.0, 0.0);
    } else if(v > FLOAT_MAX) {
      return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;
    } else if(v < -FLOAT_MAX) {
      return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;
    }

    highp vec4 c = vec4(0,0,0,0);

    highp float e = floor(log2(av));
    highp float m = exp2(fract(log2(av))) - 1.0;

    c[2] = floor(128.0 * m);
    m -= c[2] / 128.0;
    c[1] = floor(32768.0 * m);
    m -= c[1] / 32768.0;
    c[0] = floor(8388608.0 * m);

    highp float ebias = e + 127.0;
    c[3] = floor(ebias / 2.0);
    ebias -= c[3] * 2.0;
    c[2] += floor(ebias) * 128.0;

    c[3] += 128.0 * step(0.0, -v);

    return c / 255.0;
  }
`;/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const{getBroadcastDims:Yc}=rg;function vg(n,e,t){const s=[];if(n.forEach(f=>{const p=A(f.shapeInfo.logicalShape);if(f.shapeInfo.isUniform?s.push(`uniform float ${f.name}${p>1?`[${p}]`:""};`):(s.push(`uniform sampler2D ${f.name};`),s.push(`uniform int offset${f.name};`)),t.enableShapeUniforms){const{uniformShape:x}=ir(t.packedInputs,f.shapeInfo.logicalShape,f.shapeInfo.texShape);switch(x.length){case 1:s.push(`uniform int ${f.name}Shape;`);break;case 2:s.push(`uniform ivec2 ${f.name}Shape;`);break;case 3:s.push(`uniform ivec3 ${f.name}Shape;`);break;case 4:s.push(`uniform ivec4 ${f.name}Shape;`);break}s.push(`uniform ivec2 ${f.name}TexShape;`)}}),t.enableShapeUniforms){switch(e.logicalShape.length){case 1:s.push("uniform int outShape;");break;case 2:s.push("uniform ivec2 outShape;"),s.push("uniform int outShapeStrides;");break;case 3:s.push("uniform ivec3 outShape;"),s.push("uniform ivec2 outShapeStrides;");break;case 4:s.push("uniform ivec4 outShape;"),s.push("uniform ivec3 outShapeStrides;");break}s.push("uniform ivec2 outTexShape;")}t.customUniforms&&t.customUniforms.forEach(f=>{s.push(`uniform ${f.type} ${f.name}${f.arrayIndex?`[${f.arrayIndex}]`:""};`)});const o=s.join(`
`),r=n.map(f=>Sg(f,e,t.packedInputs,t.enableShapeUniforms)).join(`
`),a=e.texShape,i=ge(),c=Rg(i);let l,u,d=Ng(i);return e.isPacked?(l=Ig(e.logicalShape,a,t.enableShapeUniforms),u=kg(i)):(l=Tg(e.logicalShape,a,t.enableShapeUniforms),u=Eg(i)),t.packedInputs&&(d+=Og),[d,c,u,o,l,r,t.userCode].join(`
`)}function sn(n,e=!1){const t=n.shapeInfo.logicalShape;switch(t.length){case 0:return Xg(n,e);case 1:return jg(n,e);case 2:return Yg(n,e);case 3:return Zg(n,e);case 4:return ex(n,e);case 5:return tx(n);case 6:return nx(n);default:throw new Error(`${t.length}-D input sampling is not yet supported`)}}function Qc(n,e){switch(n.shapeInfo.logicalShape.length){case 0:return Hg(n);case 1:return qg(n,e);case 2:return Kg(n,e);case 3:return Qg(n,e);default:return Jg(n,e)}}function Sg(n,e,t=!1,s){let o="";t?o+=Qc(n,s):o+=sn(n,s);const r=n.shapeInfo.logicalShape,a=e.logicalShape;return r.length<=a.length&&(t?o+=sx(n,e):o+=ox(n,e)),o}function Ig(n,e,t){switch(n.length){case 0:return Zc();case 1:return Pg(n,e,t);case 2:return Gg(n,e,t);case 3:return Lg(n,e,t);default:return Mg(n,e,t)}}function Tg(n,e,t){switch(n.length){case 0:return Zc();case 1:return _g(n,e,t);case 2:return zg(n,e,t);case 3:return Bg(n,e,t);case 4:return Vg(n,e,t);case 5:return Ug(n,e);case 6:return Wg(n,e);default:throw new Error(`${n.length}-D output sampling is not yet supported`)}}function Rg(n){return`
    float sampleTexture(sampler2D textureSampler, vec2 uv) {
      return ${n.texture2D}(textureSampler, uv).r;
    }
  `}function Eg(n){return`
    void setOutput(float val) {
      ${n.output} = vec4(val, 0, 0, 0);
    }
  `}function kg(n){return`
    void setOutput(vec4 val) {
      ${n.output} = val;
    }
  `}function Ng(n){return`${n.version}
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    ${n.varyingFs} vec2 resultUV;
    ${n.defineOutput}
    const vec2 halfCR = vec2(0.5, 0.5);

    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    uniform float NAN;
    ${n.defineSpecialNaN}
    ${n.defineSpecialInf}
    ${n.defineRound}

    int imod(int x, int y) {
      return x - y * (x / y);
    }

    int idiv(int a, int b, float sign) {
      int res = a / b;
      int mod = imod(a, b);
      if (sign < 0. && mod != 0) {
        res -= 1;
      }
      return res;
    }

    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    #define HASHSCALE1 443.8975
    float random(float seed){
      vec2 p = resultUV * seed;
      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
      p3 += dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${Ag}
    ${Fg}
    ${Dg}
  `}const Ag=`
vec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,Fg=`
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,Dg=`
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,Og=`
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  float getChannel(vec4 frag, int dim) {
    float modCoord = mod(float(dim), 2.);
    return modCoord == 0. ? frag.r : frag.g;
  }
`;function Zc(){return`
    int getOutputCoords() {
      return 0;
    }
  `}function Pg(n,e,t){const s=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)];return s[0]===1?t?`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ceil(float(outTexShape[1]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ${s[1]}.0);
      }
    `:s[1]===1?t?`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ceil(float(outTexShape[0]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ${s[0]}.0);
      }
    `:t?`
    int getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      return 2 * (resTexRC.x * packedTexShape[1] + resTexRC.y);
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      return 2 * (resTexRC.x * ${s[1]} + resTexRC.y);
    }
  `}function _g(n,e,t){return e[0]===1?t?`
      int getOutputCoords() {
        return int(resultUV.x * float(outTexShape[1]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.x * ${e[1]}.0);
      }
    `:e[1]===1?t?`
      int getOutputCoords() {
        return int(resultUV.y * float(outTexShape[0]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.y * ${e[0]}.0);
      }
    `:t?`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      return resTexRC.x * outTexShape[1] + resTexRC.y;
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${e[0]}, ${e[1]}));
      return resTexRC.x * ${e[1]} + resTexRC.y;
    }
  `}function Lg(n,e,t){if(t)return`
    ivec3 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec3(b, r, c);
    }
  `;const s=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)],o=Math.ceil(n[2]/2),r=o*Math.ceil(n[1]/2);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      int index = resTexRC.x * ${s[1]} + resTexRC.y;

      int b = index / ${r};
      index -= b * ${r};

      int r = 2 * (index / ${o});
      int c = imod(index, ${o}) * 2;

      return ivec3(b, r, c);
    }
  `}function Bg(n,e,t){if(t)return`
  ivec3 getOutputCoords() {
    ivec2 resTexRC = ivec2(resultUV.yx *
                           vec2(outTexShape[0], outTexShape[1]));
    int index = resTexRC.x * outTexShape[1] + resTexRC.y;
    ${Rs(["r","c","d"],n)}
    return ivec3(r, c, d);
  }
`;const s=Ft(["r","c","d"],n);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${e[0]}, ${e[1]}));
      int index = resTexRC.x * ${e[1]} + resTexRC.y;
      ${s}
      return ivec3(r, c, d);
    }
  `}function Mg(n,e,t){if(t)return`
    ivec4 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int texelsInLogicalRow = int(ceil(float(outShape[3]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatchN = texelsInBatch * outShape[1];

      int b2 = index / texelsInBatchN;
      index -= b2 * texelsInBatchN;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec4(b2, b, r, c);
    }
  `;const s=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)],o=Math.ceil(n[n.length-1]/2),r=o*Math.ceil(n[n.length-2]/2);let a=r,i="",c="b, r, c";for(let l=2;l<n.length-1;l++)a*=n[n.length-l-1],i=`
      int b${l} = index / ${a};
      index -= b${l} * ${a};
    `+i,c=`b${l}, `+c;return`
    ivec${n.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      int index = resTexRC.x * ${s[1]} + resTexRC.y;

      ${i}

      int b = index / ${r};
      index -= b * ${r};

      int r = 2 * (index / ${o});
      int c = imod(index, ${o}) * 2;

      return ivec${n.length}(${c});
    }
  `}function Vg(n,e,t){if(t)return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      ${Rs(["r","c","d","d2"],n)}
      return ivec4(r, c, d, d2);
    }
  `;const s=Ft(["r","c","d","d2"],n);return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${e[0]}, ${e[1]}));
      int index = resTexRC.x * ${e[1]} + resTexRC.y;
      ${s}
      return ivec4(r, c, d, d2);
    }
  `}function Ug(n,e){const t=Ft(["r","c","d","d2","d3"],n);return`
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${e[0]},
                             ${e[1]}));

      int index = resTexRC.x * ${e[1]} + resTexRC.y;

      ${t}

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
    }
  `}function Wg(n,e){const t=Ft(["r","c","d","d2","d3","d4"],n);return`
    ivec6 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${e[0]}, ${e[1]}));
      int index = resTexRC.x * ${e[1]} + resTexRC.y;

      ${t}

      ivec6 result = ivec6(r, c, d, d2, d3, d4);
      return result;
    }
  `}function Gg(n,e,t){const s=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)];if(oe(n,e))return t?`
      ivec2 getOutputCoords() {
        ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
        return 2 * ivec2(resultUV.yx * vec2(packedTexShape[0], packedTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return 2 * ivec2(resultUV.yx * vec2(${s[0]}, ${s[1]}));
      }
    `;const o=Math.ceil(n[1]/2);return t?`
    ivec2 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));

      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;
      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));

      int index = resTexRC.x * ${s[1]} + resTexRC.y;
      int r = 2 * (index / ${o});
      int c = imod(index, ${o}) * 2;

      return ivec2(r, c);
    }
  `}function zg(n,e,t){return oe(n,e)?t?`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(outTexShape[0], outTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${e[0]}, ${e[1]}));
      }
    `:n[1]===1?t?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(index, 0);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${e[0]}, ${e[1]}));
        int index = resTexRC.x * ${e[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `:n[0]===1?t?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(0, index);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${e[0]}, ${e[1]}));
        int index = resTexRC.x * ${e[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `:t?`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      int r = index / outShape[1];
      int c = index - r * outShape[1];
      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${e[0]}, ${e[1]}));
      int index = resTexRC.x * ${e[1]} + resTexRC.y;
      int r = index / ${n[1]};
      int c = index - r * ${n[1]};
      return ivec2(r, c);
    }
  `}function Dt(n){return`offset${n}`}function Hg(n){const e=n.name,t="get"+e.charAt(0).toUpperCase()+e.slice(1),s=ge();return`
    vec4 ${t}() {
      return ${s.texture2D}(${e}, halfCR);
    }
  `}function Xg(n,e){const t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1);if(n.shapeInfo.isUniform)return`float ${s}() {return ${t};}`;const[o,r]=n.shapeInfo.texShape;if(o===1&&r===1)return`
      float ${s}() {
        return sampleTexture(${t}, halfCR);
      }
    `;const a=Dt(t);if(e)return`
    float ${s}() {
      vec2 uv = uvFromFlat(${t}TexShape[0], ${t}TexShape[1], ${a});
      return sampleTexture(${t}, uv);
    }
  `;const[i,c]=n.shapeInfo.texShape;return`
    float ${s}() {
      vec2 uv = uvFromFlat(${i}, ${c}, ${a});
      return sampleTexture(${t}, uv);
    }
  `}function qg(n,e){const t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1),o=n.shapeInfo.texShape,r=ge();if(e)return`
    vec4 ${s}(int index) {
      ivec2 packedTexShape = ivec2(ceil(float(${t}TexShape[0]) / 2.0), ceil(float(${t}TexShape[1]) / 2.0));
      vec2 uv = packedUVfrom1D(
        packedTexShape[0], packedTexShape[1], index);
      return ${r.texture2D}(${t}, uv);
    }
  `;const a=[Math.ceil(o[0]/2),Math.ceil(o[1]/2)];return`
    vec4 ${s}(int index) {
      vec2 uv = packedUVfrom1D(
        ${a[0]}, ${a[1]}, index);
      return ${r.texture2D}(${t}, uv);
    }
  `}function jg(n,e){const t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1);if(n.shapeInfo.isUniform)return`
      float ${s}(int index) {
        ${on(n)}
      }
    `;const o=n.shapeInfo.texShape,r=o[0],a=o[1];if(a===1&&r===1)return`
      float ${s}(int index) {
        return sampleTexture(${t}, halfCR);
      }
    `;const i=Dt(t);return a===1?e?`
      float ${s}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${i}) + 0.5) / float(${t}TexShape[0]));
        return sampleTexture(${t}, uv);
      }
    `:`
      float ${s}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${i}) + 0.5) / ${r}.0);
        return sampleTexture(${t}, uv);
      }
    `:r===1?e?`
      float ${s}(int index) {
        vec2 uv = vec2((float(index + ${i}) + 0.5) / float(${t}TexShape[1]), 0.5);
        return sampleTexture(${t}, uv);
      }
    `:`
      float ${s}(int index) {
        vec2 uv = vec2((float(index + ${i}) + 0.5) / ${a}.0, 0.5);
        return sampleTexture(${t}, uv);
      }
    `:e?`
    float ${s}(int index) {
      vec2 uv = uvFromFlat(${t}TexShape[0], ${t}TexShape[1], index + ${i});
      return sampleTexture(${t}, uv);
    }
  `:`
    float ${s}(int index) {
      vec2 uv = uvFromFlat(${r}, ${a}, index + ${i});
      return sampleTexture(${t}, uv);
    }
  `}function Kg(n,e){const t=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=n.shapeInfo.texShape,a=r[0],i=r[1],c=ge();if(r!=null&&oe(t,r))return e?`
      vec4 ${o}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);

        return ${c.texture2D}(${s}, uv);
      }
    `:`
      vec4 ${o}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${i}.0, ${a}.0);

        return ${c.texture2D}(${s}, uv);
      }
    `;if(e)return`
    vec4 ${o}(int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${s}TexShape[0]) / 2.0), ceil(float(${s}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${s}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom2D(valuesPerRow, packedTexShape[0], packedTexShape[1], row, col);
      return ${c.texture2D}(${s}, uv);
    }
  `;const l=[Math.ceil(r[0]/2),Math.ceil(r[1]/2)],u=Math.ceil(t[1]/2);return`
    vec4 ${o}(int row, int col) {
      vec2 uv = packedUVfrom2D(${u}, ${l[0]}, ${l[1]}, row, col);
      return ${c.texture2D}(${s}, uv);
    }
  `}function Yg(n,e){const t=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=n.shapeInfo.texShape;if(r!=null&&oe(t,r)){if(e)return`
      float ${o}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `;const h=r[0],f=r[1];return`
    float ${o}(int row, int col) {
      vec2 uv = (vec2(col, row) + halfCR) / vec2(${f}.0, ${h}.0);
      return sampleTexture(${s}, uv);
    }
  `}const{newShape:a,keptDims:i}=lt(t),c=a;if(c.length<t.length){const h=rn(n,c),f=["row","col"];return`
      ${sn(h,e)}
      float ${o}(int row, int col) {
        return ${o}(${an(f,i)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${o}(int row, int col) {
        int index = round(dot(vec2(row, col), vec2(${t[1]}, 1)));
        ${on(n)}
      }
    `;const l=r[0],u=r[1],d=Dt(s);return u===1?e?`
      float ${o}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${s}Shape[1], 1, 1));
        vec2 uv = vec2(0.5, (index + 0.5) / float(${s}TexShape[0]));
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${o}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${t[1]}, 1, 1));
      vec2 uv = vec2(0.5, (index + 0.5) / ${l}.0);
      return sampleTexture(${s}, uv);
    }
  `:l===1?e?`
      float ${o}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${s}Shape[1], 1, 1));
        vec2 uv = vec2((index + 0.5) / float(${s}TexShape[1]), 0.5);
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${o}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${t[1]}, 1, 1));
      vec2 uv = vec2((index + 0.5) / ${u}.0, 0.5);
      return sampleTexture(${s}, uv);
    }
  `:e?`
      float ${o}(int row, int col) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${s}Shape[1] + col + ${d};
        vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index);
        return sampleTexture(${s}, uv);
      }
    `:`
  float ${o}(int row, int col) {
    // Explicitly use integer operations as dot() only works on floats.
    int index = row * ${t[1]} + col + ${d};
    vec2 uv = uvFromFlat(${l}, ${u}, index);
    return sampleTexture(${s}, uv);
  }
`}function Qg(n,e){const t=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=n.shapeInfo.texShape,a=[Math.ceil(r[0]/2),Math.ceil(r[1]/2)];if(t[0]===1){const h=t.slice(1),f=[1,2],p=rn(n,h),x=["b","row","col"];return`
        ${Qc(p,e)}
        vec4 ${o}(int b, int row, int col) {
          return ${o}(${an(x,f)});
        }
      `}const i=ge();if(e)return`
    vec4 ${o}(int b, int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${s}TexShape[0]) / 2.0), ceil(float(${s}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${s}Shape[2]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${s}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom3D(
        packedTexShape[0], packedTexShape[1], texelsInBatch, valuesPerRow, b, row, col);
      return ${i.texture2D}(${s}, uv);
    }
  `;const c=a[0],l=a[1],u=Math.ceil(t[2]/2),d=u*Math.ceil(t[1]/2);return`
    vec4 ${o}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${c}, ${l}, ${d}, ${u}, b, row, col);
      return ${i.texture2D}(${s}, uv);
    }
  `}function Zg(n,e){const t=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=t[1]*t[2],a=t[2],{newShape:i,keptDims:c}=lt(t),l=i;if(l.length<t.length){const x=rn(n,l),g=["row","col","depth"];return`
        ${sn(x,e)}
        float ${o}(int row, int col, int depth) {
          return ${o}(${an(g,c)});
        }
      `}if(n.shapeInfo.isUniform)return`
      float ${o}(int row, int col, int depth) {
        int index = round(dot(vec3(row, col, depth),
                          vec3(${r}, ${a}, 1)));
        ${on(n)}
      }
    `;const u=n.shapeInfo.texShape,d=u[0],h=u[1],f=n.shapeInfo.flatOffset;if(h===r&&f==null)return e?`
      float ${o}(int row, int col, int depth) {
        int stride1 = ${s}Shape[2];
        float texR = float(row);
        float texC = dot(vec2(col, depth), vec2(stride1, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
        float ${o}(int row, int col, int depth) {
          float texR = float(row);
          float texC = dot(vec2(col, depth), vec2(${a}, 1));
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${h}.0, ${d}.0);
          return sampleTexture(${s}, uv);
        }
      `;if(h===a&&f==null)return e?`
      float ${o}(int row, int col, int depth) {
        float texR = dot(vec2(row, col), vec2(${s}Shape[1], 1));
        float texC = float(depth);
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${o}(int row, int col, int depth) {
      float texR = dot(vec2(row, col), vec2(${t[1]}, 1));
      float texC = float(depth);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${h}.0, ${d}.0);
      return sampleTexture(${s}, uv);
    }
  `;const p=Dt(s);return e?`
    float ${o}(int row, int col, int depth) {
      // Explicitly use integer operations as dot() only works on floats.
      int stride0 = ${s}Shape[1] * ${s}Shape[2];
      int stride1 = ${s}Shape[2];
      int index = row * stride0 + col * stride1 + depth + ${p};
      vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index);
      return sampleTexture(${s}, uv);
    }
    `:`
      float ${o}(int row, int col, int depth) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${r} + col * ${a} + depth + ${p};
        vec2 uv = uvFromFlat(${d}, ${h}, index);
        return sampleTexture(${s}, uv);
      }
  `}function Jg(n,e){const t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1),o=ge();if(e)return`
    vec4 ${s}(int b2, int b, int row, int col) {
      int valuesPerRow = int(ceil(float(${t}Shape[3]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${t}Shape[2]) / 2.0));
      int index = b * texelsInBatch + (row / 2) * valuesPerRow + (col / 2);
      texelsInBatch *= ${t}Shape[1];
      index = b2 * texelsInBatch + index;
      ivec2 packedTexShape = ivec2(ceil(float(${t}TexShape[0]) / 2.0), ceil(float(${t}TexShape[1]) / 2.0));
      int texR = index / packedTexShape[1];
      int texC = index - texR * packedTexShape[1];
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(packedTexShape[1], packedTexShape[0]); return ${o.texture2D}(${t}, uv);
    }
  `;const r=n.shapeInfo.logicalShape,a=r.length,i=n.shapeInfo.texShape,c=[Math.ceil(i[0]/2),Math.ceil(i[1]/2)],l=c[0],u=c[1],d=Math.ceil(r[a-1]/2);let h=d*Math.ceil(r[a-2]/2),f="int b, int row, int col",p=`b * ${h} + (row / 2) * ${d} + (col / 2)`;for(let x=2;x<a-1;x++)f=`int b${x}, `+f,h*=r[a-x-1],p=`b${x} * ${h} + `+p;return`
    vec4 ${s}(${f}) {
      int index = ${p};
      int texR = index / ${u};
      int texC = index - texR * ${u};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${u}, ${l});
      return ${o.texture2D}(${t}, uv);
    }
  `}function ex(n,e){const t=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=t[3],a=t[2]*r,i=t[1]*a,{newShape:c,keptDims:l}=lt(t);if(c.length<t.length){const b=rn(n,c),C=["row","col","depth","depth2"];return`
      ${sn(b,e)}
      float ${o}(int row, int col, int depth, int depth2) {
        return ${o}(${an(C,l)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${o}(int row, int col, int depth, int depth2) {
        int index = round(dot(vec4(row, col, depth, depth2),
                          vec4(${i}, ${a}, ${r}, 1)));
        ${on(n)}
      }
    `;const u=n.shapeInfo.flatOffset,d=n.shapeInfo.texShape,h=d[0],f=d[1],p=`int stride2 = ${s}Shape[3];`,x=`int stride1 = ${s}Shape[2] * stride2;`,g=`int stride0 = ${s}Shape[1] * stride1;`;if(f===i&&u==null)return e?`
      float ${o}(int row, int col, int depth, int depth2) {
        ${p}
        ${x}
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(stride1, stride2, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
      float ${o}(int row, int col, int depth, int depth2) {
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(${a}, ${r}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${f}.0, ${h}.0);
        return sampleTexture(${s}, uv);
      }
    `;if(f===r&&u==null)return e?`
      float ${o}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${s}Shape[1] * ${s}Shape[2], ${s}Shape[2], 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
      float ${o}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${t[1]*t[2]}, ${t[2]}, 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${f}.0, ${h}.0);
        return sampleTexture(${s}, uv);
      }
    `;const m=Dt(s);return e?`
    float ${o}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      ${p}
      ${x}
      ${g}
      int index = row * stride0 + col * stride1 +
          depth * stride2 + depth2;
      vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index + ${m});
      return sampleTexture(${s}, uv);
    }
  `:`
    float ${o}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${i} + col * ${a} +
          depth * ${r} + depth2;
      vec2 uv = uvFromFlat(${h}, ${f}, index + ${m});
      return sampleTexture(${s}, uv);
    }
  `}function tx(n){const e=n.shapeInfo.logicalShape,t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1),o=e[4],r=e[3]*o,a=e[2]*r,i=e[1]*a,{newShape:c,keptDims:l}=lt(e);if(c.length<e.length){const x=rn(n,c),g=["row","col","depth","depth2","depth3"];return`
      ${sn(x)}
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        return ${s}(${an(g,l)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${i}, ${a}, ${r}, ${o})) +
          depth3;
        ${on(n)}
      }
    `;const u=n.shapeInfo.flatOffset,d=n.shapeInfo.texShape,h=d[0],f=d[1];if(f===i&&u==null)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
                         vec4(${a}, ${r}, ${o}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${f}.0, ${h}.0);
        return sampleTexture(${t}, uv);
      }
    `;if(f===o&&u==null)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${e[1]*e[2]*e[3]},
               ${e[2]*e[3]}, ${e[3]}, 1));
        int texC = depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${f}.0, ${h}.0);
        return sampleTexture(${t}, uv);
      }
    `;const p=Dt(t);return`
    float ${s}(int row, int col, int depth, int depth2, int depth3) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${i} + col * ${a} + depth * ${r} +
          depth2 * ${o} + depth3 + ${p};
      vec2 uv = uvFromFlat(${h}, ${f}, index);
      return sampleTexture(${t}, uv);
    }
  `}function nx(n){const e=n.shapeInfo.logicalShape,t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1),{newShape:o,keptDims:r}=lt(e);if(o.length<e.length){const g=rn(n,o),m=["row","col","depth","depth2","depth3","depth4"];return`
      ${sn(g)}
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        return ${s}(${an(m,r)});
      }
    `}const a=e[5],i=e[4]*a,c=e[3]*i,l=e[2]*c,u=e[1]*l;if(n.shapeInfo.isUniform)return`
      float ${s}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
        int index = round(dot(
          vec4(row, col, depth, depth2),
          vec4(${u}, ${l}, ${c}, ${i})) +
          dot(
            vec2(depth3, depth4),
            vec2(${a}, 1)));
        ${on(n)}
      }
    `;const d=n.shapeInfo.flatOffset,h=n.shapeInfo.texShape,f=h[0],p=h[1];if(p===u&&d==null)return`
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
          vec4(${l}, ${c}, ${i}, ${a})) +
               float(depth4);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${p}.0, ${f}.0);
        return sampleTexture(${t}, uv);
      }
    `;if(p===a&&d==null)return`
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        float texR = dot(vec4(row, col, depth, depth2),
          vec4(${e[1]*e[2]*e[3]*e[4]},
               ${e[2]*e[3]*e[4]},
               ${e[3]*e[4]},
               ${e[4]})) + float(depth3);
        int texC = depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${p}.0, ${f}.0);
        return sampleTexture(${t}, uv);
      }
    `;const x=Dt(t);return`
    float ${s}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${u} + col * ${l} + depth * ${c} +
          depth2 * ${i} + depth3 * ${a} + depth4 + ${x};
      vec2 uv = uvFromFlat(${f}, ${p}, index);
      return sampleTexture(${t}, uv);
    }
  `}function on(n){const e=n.name,t=A(n.shapeInfo.logicalShape);return t<2?`return ${e};`:`
    for (int i = 0; i < ${t}; i++) {
      if (i == index) {
        return ${e}[i];
      }
    }
  `}function sx(n,e){const t=n.name,s=t.charAt(0).toUpperCase()+t.slice(1),o="get"+s+"AtOutCoords",r=n.shapeInfo.logicalShape.length,a=e.logicalShape.length,i=Yc(n.shapeInfo.logicalShape,e.logicalShape),c=H(a),l=a-r;let u;const d=["x","y","z","w","u","v"];r===0?u="":a<2&&i.length>=1?u="coords = 0;":u=i.map(b=>`coords.${d[b+l]} = 0;`).join(`
`);let h="";a<2&&r>0?h="coords":h=n.shapeInfo.logicalShape.map((b,C)=>`coords.${d[C+l]}`).join(", ");let f="return outputValue;";const x=A(n.shapeInfo.logicalShape)===1,m=A(e.logicalShape)===1;if(r===1&&!x&&!m)f=`
      return vec4(outputValue.xy, outputValue.xy);
    `;else if(x&&!m)a===1?f=`
        return vec4(outputValue.x, outputValue.x, 0., 0.);
      `:f=`
        return vec4(outputValue.x);
      `;else if(i.length){const b=r-2,C=r-1;i.indexOf(b)>-1&&i.indexOf(C)>-1?f="return vec4(outputValue.x);":i.indexOf(b)>-1?f="return vec4(outputValue.x, outputValue.y, outputValue.x, outputValue.y);":i.indexOf(C)>-1&&(f="return vec4(outputValue.xx, outputValue.zz);")}return`
    vec4 ${o}() {
      ${c} coords = getOutputCoords();
      ${u}
      vec4 outputValue = get${s}(${h});
      ${f}
    }
  `}function ox(n,e){const t=n.name,s=t.charAt(0).toUpperCase()+t.slice(1),o="get"+s+"AtOutCoords",r=e.texShape,a=n.shapeInfo.texShape,i=n.shapeInfo.logicalShape.length,c=e.logicalShape.length;if(!n.shapeInfo.isUniform&&i===c&&n.shapeInfo.flatOffset==null&&oe(a,r))return`
      float ${o}() {
        return sampleTexture(${t}, resultUV);
      }
    `;const l=H(c),u=Yc(n.shapeInfo.logicalShape,e.logicalShape),d=c-i;let h;const f=["x","y","z","w","u","v"];i===0?h="":c<2&&u.length>=1?h="coords = 0;":h=u.map(x=>`coords.${f[x+d]} = 0;`).join(`
`);let p="";return c<2&&i>0?p="coords":p=n.shapeInfo.logicalShape.map((x,g)=>`coords.${f[g+d]}`).join(", "),`
    float ${o}() {
      ${l} coords = getOutputCoords();
      ${h}
      return get${s}(${p});
    }
  `}function H(n){if(n<=1)return"int";if(n===2)return"ivec2";if(n===3)return"ivec3";if(n===4)return"ivec4";if(n===5)return"ivec5";if(n===6)return"ivec6";throw Error(`GPU for rank ${n} is not yet supported`)}function ir(n,e,t){const{newShape:s,keptDims:o}=lt(e),r=e.length,a=n&&r===3&&e[0]===1,i=a?e.slice(1):s,c=!n&&r>1&&!oe(e,t)&&s.length<r||a;return{useSqueezeShape:c,uniformShape:c?i:e,keptDims:o}}function rn(n,e){const t=JSON.parse(JSON.stringify(n));return t.shapeInfo.logicalShape=e,t}function an(n,e){return e.map(t=>n[t]).join(", ")}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rx(n,e,t,s){const o=t.map((u,d)=>{const h={logicalShape:u.shape,texShape:u.isUniform?null:u.texData.texShape,isUniform:u.isUniform,isPacked:u.isUniform?!1:u.texData.isPacked,flatOffset:null};return u.texData!=null&&u.texData.slice!=null&&u.texData.slice.flatOffset>0&&(h.flatOffset=u.texData.slice.flatOffset),{name:e.variableNames[d],shapeInfo:h}}),r=o.map(u=>u.shapeInfo),a={logicalShape:s.shape,texShape:s.texData.texShape,isUniform:!1,isPacked:s.texData.isPacked,flatOffset:null},i=vg(o,a,e),c=Ec(n.gl,i),l=n.createProgram(c);return $().get("ENGINE_COMPILE_ONLY")?{program:e,fragmentShader:c,source:i,webGLProgram:l,inShapeInfos:r,outShapeInfo:a,variablesLocations:null,customUniformLocations:null,infLoc:null,nanLoc:null,outShapeLocation:null,outShapeStridesLocation:null,outTexShapeLocation:null}:(n.buildVao(l),Object.assign({program:e,fragmentShader:c,source:i,webGLProgram:l,inShapeInfos:r,outShapeInfo:a},Jc(n,e,l)))}function Jc(n,e,t){const s=[],o=[];let r,a,i,c=null,l=null;l=n.getUniformLocation(t,"NAN",!1),$().getNumber("WEBGL_VERSION")===1&&(c=n.getUniformLocation(t,"INFINITY",!1));const u=!1;for(const d of e.variableNames){const h={name:d,uniform:n.getUniformLocation(t,d,u),offset:n.getUniformLocation(t,`offset${d}`,u)};e.enableShapeUniforms&&(h.shape=n.getUniformLocation(t,`${d}Shape`,u),h.texShape=n.getUniformLocation(t,`${d}TexShape`,u)),s.push(h)}if(e.enableShapeUniforms&&(r=n.getUniformLocation(t,"outShape",u),i=n.getUniformLocation(t,"outShapeStrides",u),a=n.getUniformLocation(t,"outTexShape",u)),e.customUniforms)for(const d of e.customUniforms)o.push(n.getUniformLocation(t,d.name,u));return{variablesLocations:s,customUniformLocations:o,infLoc:c,nanLoc:l,outShapeLocation:r,outShapeStridesLocation:i,outTexShapeLocation:a}}function Wr(n,e){if(n.length!==e.length)throw Error(`Binary was compiled with ${n.length} inputs, but was executed with ${e.length} inputs`);n.forEach((t,s)=>{const o=t.logicalShape,r=e[s],a=r.shape;if(!oe(o,a))throw Error(`Binary was compiled with different shapes than the current args. Shapes ${o} and ${a} must match`);if(t.isUniform&&r.isUniform)return;const i=t.texShape,c=r.isUniform?null:r.texData.texShape;if(!oe(i,c))throw Error(`Binary was compiled with different texture shapes than the current args. Shape ${i} and ${c} must match`)})}function ax(n,e,t,s,o){e.program.enableShapeUniforms||(Wr(e.inShapeInfos,t),Wr([e.outShapeInfo],[s]));const r=s.texData.texture,a=s.texData.texShape;s.texData.isPacked?n.setOutputPackedMatrixTexture(r.texture,a[0],a[1]):n.setOutputMatrixTexture(r.texture,a[0],a[1]),n.setProgram(e.webGLProgram),n.bindVertexArray(e.webGLProgram.vao),$().getNumber("WEBGL_VERSION")===1&&e.infLoc!==null&&n.gl.uniform1f(e.infLoc,1/0),e.nanLoc!==null&&n.gl.uniform1f(e.nanLoc,NaN);for(let c=0;c<t.length;++c){const l=t[c],{uniform:u,offset:d,shape:h,texShape:f}=e.variablesLocations[c];if(h){const{uniformShape:p}=ir(e.program.packedInputs,l.shape,l.texData.texShape);switch(p.length){case 1:n.gl.uniform1iv(h,new Int32Array(p));break;case 2:n.gl.uniform2iv(h,new Int32Array(p));break;case 3:n.gl.uniform3iv(h,new Int32Array(p));break;case 4:n.gl.uniform4iv(h,new Int32Array(p));break}}if(f&&n.gl.uniform2i(f,l.texData.texShape[0],l.texData.texShape[1]),u!=null){if(l.isUniform){if(A(l.shape)<2)n.gl.uniform1f(u,l.uniformValues[0]);else{let p=l.uniformValues;p instanceof Float32Array||(p=new Float32Array(p)),n.gl.uniform1fv(u,p)}continue}l.texData.slice!=null&&d!=null&&n.gl.uniform1i(d,l.texData.slice.flatOffset),n.setInputMatrixTexture(l.texData.texture.texture,u,c)}}const i=e.outShapeLocation;if(i)switch(s.shape.length){case 1:n.gl.uniform1iv(i,new Int32Array(s.shape));break;case 2:n.gl.uniform2iv(i,new Int32Array(s.shape));break;case 3:n.gl.uniform3iv(i,new Int32Array(s.shape));break;case 4:n.gl.uniform4iv(i,new Int32Array(s.shape));break}if(e.outShapeStridesLocation){const c=J(s.shape);switch(s.shape.length){case 2:n.gl.uniform1iv(e.outShapeStridesLocation,new Int32Array(c));break;case 3:n.gl.uniform2iv(e.outShapeStridesLocation,new Int32Array(c));break;case 4:n.gl.uniform3iv(e.outShapeStridesLocation,new Int32Array(c));break}}if(e.outTexShapeLocation&&n.gl.uniform2i(e.outTexShapeLocation,s.texData.texShape[0],s.texData.texShape[1]),e.program.customUniforms&&o)for(let c=0;c<e.program.customUniforms.length;++c){const l=e.program.customUniforms[c],u=e.customUniformLocations[c],d=o[c];if(l.type==="float")n.gl.uniform1fv(u,d);else if(l.type==="vec2")n.gl.uniform2fv(u,d);else if(l.type==="vec3")n.gl.uniform3fv(u,d);else if(l.type==="vec4")n.gl.uniform4fv(u,d);else if(l.type==="int")n.gl.uniform1iv(u,d);else if(l.type==="ivec2")n.gl.uniform2iv(u,d);else if(l.type==="ivec3")n.gl.uniform3iv(u,d);else if(l.type==="ivec4")n.gl.uniform4iv(u,d);else throw Error(`uniform type ${l.type} is not supported yet.`)}n.executeProgram()}function ix(n,e,t){let s="";e.concat(t).forEach(a=>{const i=a.texData!=null&&a.texData.slice!=null&&a.texData.slice.flatOffset>0;if(n.enableShapeUniforms&&!a.isUniform){const c=a.texData.texShape,{useSqueezeShape:l,uniformShape:u,keptDims:d}=ir(n.packedInputs,a.shape,c);let h="",f="",p="";if(u.length===1&&n.packedInputs){const S=[Math.ceil(c[0]/2),Math.ceil(c[1]/2)];h=`${S[0]>1}_${S[1]>1}`}else if(u.length===2&&!n.packedInputs)f=`${u[0]>1}_${u[1]>1}`;else if(u.length>2&&!n.packedInputs){const S=J(u);p=`${S[0]===c[1]}_${S[S.length-1]===c[1]}`}const x=a.shape.length,g=u.length===2&&oe(a.shape,c),m=A(a.shape)===1,b=St(a.shape,t.shape),C=!n.packedInputs&&x===t.shape.length&&oe(c,t.texData.texShape),y=n.packedInputs||u.length>2?"":`${c[0]>1}_${c[1]>1}`;s+=`${x}_${C}_${l?d:""}_${u.length}_${m}_${b}_${g}_${h}_${f}_${p}_${y}_${i}`}else{const c=a.isUniform?"uniform":a.texData.texShape;s+=`${a.shape}_${c}_${i}`}});const o=n.userCode;let r=n.constructor.name;return r+="_"+s+"_"+o+`${$().getNumber("WEBGL_VERSION")}`,r}function de(n){return $().getBool("WEBGL_USE_SHAPES_UNIFORMS")&&n<=4}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class cx{constructor(e){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outPackingScheme=Tn.DENSE,this.customUniforms=[{name:"texShape",type:"ivec2"}];const t=ge();this.outputShape=e,this.enableShapeUniforms=de(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?Rs(["r","c","d"],e):Ft(["r","c","d"],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getA(rc.x, rc.y, rc.z);
        }

        ${t.output} = result;
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class lx{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outPackingScheme=Tn.DENSE,this.customUniforms=[{name:"texShape",type:"ivec2"}];const t=ge();this.outputShape=e,this.enableShapeUniforms=de(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?Rs(["r","c","d"],e):Ft(["r","c","d"],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getChannel(getA(rc.x, rc.y, rc.z), vec2(rc.y, rc.z));
        }

        ${t.output} = result;
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ux{constructor(e){this.variableNames=["A"],this.outTexUsage=ve.DOWNLOAD;const t=ge();this.outputShape=e,this.userCode=`
      ${Kc}

      void main() {
        float x = getAAtOutCoords();
        ${t.output} = encode_float(x);
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class dx{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outTexUsage=ve.DOWNLOAD;const t=ge();this.outputShape=e,this.userCode=`
      ${Kc}

      void main() {
        ivec3 coords = getOutputCoords();
        float x = getChannel(getAAtOutCoords(), vec2(coords.y, coords.z));
        ${t.output} = encode_float(x);
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hx={R:0,G:1,B:2,A:3};class Gr{constructor(e,t=!1,s="RGBA"){this.variableNames=["A"],this.customUniforms=[{name:"texShape",type:"ivec2"}];const o=ge();this.outputShape=e,this.enableShapeUniforms=de(this.outputShape.length);let r="result";t&&(r="floor(result * 255. + 0.5)");let a="";for(let i=0;i<s.length;i++){const c=s[i];a+=`
          if(offset == ${i}) {
            result = values[${hx[c]}];
          }`}this.userCode=`
      ${this.enableShapeUniforms?ar():rr(e)}

      void main() {
        ivec3 coords = getOutputCoords();
        int flatIndex = getFlatIndex(coords);
        float result = 0.;
        int offset = imod(flatIndex, ${s.length});

        flatIndex = idiv(flatIndex, ${s.length}, 1.);

        int r = flatIndex / texShape[1];
        if (r < texShape[0]) {
          int c = imod(flatIndex, texShape[1]);
          vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
          vec4 values = ${o.texture2D}(A, uv);
          ${a}
        }
        ${o.output} = vec4(${r}, 0., 0., 0.);
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class fx{constructor(e,t=!1){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.customUniforms=[{name:"texShape",type:"ivec2"}];const s=ge();this.outputShape=e,this.enableShapeUniforms=de(this.outputShape.length);let o="",r="result";t&&(r="floor(result * 255. + 0.5)");for(let a=0;a<=1;a++)for(let i=0;i<=1;i++){const c=a*2+i;o+=`
          localCoords = coords;
          if(localCoords[2] + ${i} < ${this.enableShapeUniforms?"outShape[2]":`${e[2]}`}) {
          localCoords[2] += ${i};
          if (localCoords[1] + ${a} < ${this.enableShapeUniforms?"outShape[1]":`${e[1]}`}) {
            localCoords[1] += ${a};

            flatIndex = getFlatIndex(localCoords);
            offset = imod(flatIndex, 4);

            flatIndex = idiv(flatIndex, 4, 1.);

            int r = flatIndex / texShape[1];
            int c = imod(flatIndex, texShape[1]);
            vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
            values = ${s.texture2D}(A, uv);

            if (offset == 0) {
              result[${c}] = values[0];
            } else if (offset == 1) {
              result[${c}] = values[1];
            } else if (offset == 2) {
              result[${c}] = values[2];
            } else {
              result[${c}] = values[3];
            }
          }
        }
        `}this.userCode=`
        ${this.enableShapeUniforms?ar():rr(e)}

        void main() {
          ivec3 coords = getOutputCoords();

          vec4 result = vec4(0.);
          int flatIndex, r, c, offset;
          ivec3 localCoords;
          vec2 uv;
          vec4 values;

          ${o}

          ${s.output} = ${r};
        }
    `}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function el(n){const e=ge(),t=`${e.version}
    precision highp float;
    ${e.attribute} vec3 clipSpacePos;
    ${e.attribute} vec2 uv;
    ${e.varyingVs} vec2 resultUV;

    void main() {
      gl_Position = vec4(clipSpacePos, 1);
      resultUV = uv;
    }`;return Rc(n,t)}function tl(n){const e=new Float32Array([-1,1,0,0,1,-1,-1,0,0,0,1,1,0,1,1,1,-1,0,1,0]);return Ac(n,e)}function nl(n){const e=new Uint16Array([0,1,2,2,1,3]);return Fc(n,e)}function Un(n,e,t,s,o,r){Oc(e,t);const a=Dc(n),i=n.TEXTURE_2D;return D(n,()=>n.bindTexture(i,a)),D(n,()=>n.texParameteri(i,n.TEXTURE_WRAP_S,n.CLAMP_TO_EDGE)),D(n,()=>n.texParameteri(i,n.TEXTURE_WRAP_T,n.CLAMP_TO_EDGE)),D(n,()=>n.texParameteri(i,n.TEXTURE_MIN_FILTER,n.NEAREST)),D(n,()=>n.texParameteri(i,n.TEXTURE_MAG_FILTER,n.NEAREST)),$().getNumber("WEBGL_VERSION")===1?D(n,()=>n.texImage2D(i,0,s,e,t,0,o,r,null)):D(n,()=>n.texStorage2D(i,1,s,e,t)),D(n,()=>n.bindTexture(n.TEXTURE_2D,null)),{texture:a,texShape:[t,e]}}function cr(n){return n.internalFormatFloat}function sl(n,e,t,s){const[o,r]=Vn(e,t);return Un(n,o,r,cr(s),s.textureFormatFloat,n.FLOAT)}function lr(n){return n.internalFormatHalfFloat}function ol(n,e,t,s){const[o,r]=Vn(e,t);return Un(n,o,r,lr(s),s.textureFormatFloat,s.textureTypeHalfFloat)}function ur(n){return n.downloadTextureFormat}function rl(n,e,t,s){const[o,r]=Vn(e,t);return Un(n,o,r,ur(s),n.RGBA,n.UNSIGNED_BYTE)}function dr(n){return n.internalFormatPackedFloat}function al(n,e,t,s){const[o,r]=tn(e,t);return Un(n,o,r,dr(s),n.RGBA,n.FLOAT)}function hr(n){return n.internalFormatPackedHalfFloat}function il(n,e,t,s){const[o,r]=tn(e,t);return Un(n,o,r,hr(s),n.RGBA,s.textureTypeHalfFloat)}function cl(n,e,t){return D(n,()=>n.bindBuffer(n.ARRAY_BUFFER,t)),oo(n,e,"clipSpacePos",t,3,20,0)&&oo(n,e,"uv",t,2,20,12)}function ll(n,e,t,s,o,r){D(n,()=>n.bindTexture(n.TEXTURE_2D,e));let a,i,c;o instanceof Uint8Array?(a=new Uint8Array(t*s*4),i=n.UNSIGNED_BYTE,c=n.RGBA):(a=new Float32Array(t*s*4),i=n.FLOAT,c=r.internalFormatPackedFloat),a.set(o),$().getNumber("WEBGL_VERSION")===2?D(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,t,s,n.RGBA,i,a)):D(n,()=>n.texImage2D(n.TEXTURE_2D,0,c,t,s,0,n.RGBA,i,a)),D(n,()=>n.bindTexture(n.TEXTURE_2D,null))}function ul(n,e,t){D(n,()=>n.bindTexture(n.TEXTURE_2D,e)),t.data instanceof Uint8Array?$().getNumber("WEBGL_VERSION")===2?D(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,t.width,t.height,n.RGBA,n.UNSIGNED_BYTE,t.data)):D(n,()=>n.texImage2D(n.TEXTURE_2D,0,n.RGBA,t.width,t.height,0,n.RGBA,n.UNSIGNED_BYTE,t.data)):$().getNumber("WEBGL_VERSION")===2?D(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,n.RGBA,n.UNSIGNED_BYTE,t)):D(n,()=>n.texImage2D(n.TEXTURE_2D,0,n.RGBA,n.RGBA,n.UNSIGNED_BYTE,t)),D(n,()=>n.bindTexture(n.TEXTURE_2D,null))}function dl(n,e,t,s){const o=n.createBuffer();D(n,()=>n.bindBuffer(n.PIXEL_PACK_BUFFER,o));const i=4*4*e*t;return D(n,()=>n.bufferData(n.PIXEL_PACK_BUFFER,i,n.STREAM_READ)),D(n,()=>n.readPixels(0,0,t,e,n.RGBA,n.FLOAT,0)),D(n,()=>n.bindBuffer(n.PIXEL_PACK_BUFFER,null)),o}function hl(n,e,t){const s=n,o=new Float32Array(t);return s.bindBuffer(s.PIXEL_PACK_BUFFER,e),s.getBufferSubData(s.PIXEL_PACK_BUFFER,0,o),s.bindBuffer(s.PIXEL_PACK_BUFFER,null),o}function fl(n,e,t,s){const[o,r]=Vn(e,t),a=4,i=new Uint8Array(cg(e*t,a));return D(n,()=>n.readPixels(0,0,o,r,s.downloadTextureFormat,n.UNSIGNED_BYTE,i)),new Float32Array(i.buffer)}function pl(n,e,t,s,o,r,a,i){const c=n,l=new Float32Array(lg(r,a));return c.bindBuffer(c.PIXEL_PACK_BUFFER,e),c.getBufferSubData(c.PIXEL_PACK_BUFFER,0,l),c.bindBuffer(c.PIXEL_PACK_BUFFER,null),l}function ml(n,e,t){const s=new Float32Array(e*t*4);return D(n,()=>n.readPixels(0,0,t,e,n.RGBA,n.FLOAT,s)),s}const px=Object.freeze(Object.defineProperty({__proto__:null,bindVertexProgramAttributeStreams:cl,createBufferFromOutputTexture:dl,createFloat16MatrixTexture:ol,createFloat16PackedMatrixTexture:il,createFloat32MatrixTexture:sl,createIndexBuffer:nl,createPackedMatrixTexture:al,createUnsignedBytesMatrixTexture:rl,createVertexBuffer:tl,createVertexShader:el,downloadByteEncodedFloatMatrixFromOutputTexture:fl,downloadFloat32MatrixFromBuffer:hl,downloadMatrixFromPackedOutputTexture:ml,downloadPackedMatrixFromBuffer:pl,getInternalFormatForFloat16MatrixTexture:lr,getInternalFormatForFloat16PackedMatrixTexture:hr,getInternalFormatForFloat32MatrixTexture:cr,getInternalFormatForPackedMatrixTexture:dr,getInternalFormatForUnsignedBytesMatrixTexture:ur,uploadDenseMatrixToTexture:ll,uploadPixelDataToTexture:ul},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ss{constructor(e){this.outputTexture=null,this.program=null,this.disposed=!1,this.itemsToPoll=[];const t=$().getNumber("WEBGL_VERSION");if(e!=null?(this.gl=e,Sc(t,e)):this.gl=Pe(t),e=this.gl,$().getNumber("WEBGL_VERSION")===2){const r=e;this.createVertexArray=()=>D(r,()=>r.createVertexArray()),this.bindVertexArray=a=>D(r,()=>r.bindVertexArray(a)),this.deleteVertexArray=a=>D(r,()=>r.deleteVertexArray(a)),this.getVertexArray=()=>D(r,()=>r.getParameter(r.VERTEX_ARRAY_BINDING))}else if(e!=null){const r=e.getExtension("OES_vertex_array_object");if(r==null)throw new Error("All WebGL1 implementations are expected to offer OES_vertex_array_object.");this.createVertexArray=()=>D(e,()=>r.createVertexArrayOES()),this.bindVertexArray=a=>D(e,()=>r.bindVertexArrayOES(a)),this.deleteVertexArray=a=>D(e,()=>r.deleteVertexArrayOES(a)),this.getVertexArray=()=>D(e,()=>e.getParameter(r.VERTEX_ARRAY_BINDING_OES))}let s="WEBGL_color_buffer_float";const o="EXT_color_buffer_half_float";if(this.parallelCompilationExtension=this.gl.getExtension("KHR_parallel_shader_compile"),$().getNumber("WEBGL_VERSION")===1){const r="OES_texture_float",a="OES_texture_half_float";if(this.textureFloatExtension=pn(this.gl,r),Se(this.gl,a))this.textureHalfFloatExtension=pn(this.gl,a);else if($().get("WEBGL_FORCE_F16_TEXTURES"))throw new Error("GL context does not support half float textures, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.");if(this.colorBufferFloatExtension=this.gl.getExtension(s),Se(this.gl,o))this.colorBufferHalfFloatExtension=pn(this.gl,o);else if($().get("WEBGL_FORCE_F16_TEXTURES"))throw new Error("GL context does not support color renderable half floats, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.")}else if(s="EXT_color_buffer_float",Se(this.gl,s))this.colorBufferFloatExtension=this.gl.getExtension(s);else if(Se(this.gl,o))this.colorBufferHalfFloatExtension=this.gl.getExtension(o);else throw new Error("GL context does not support color renderable floats");this.vertexBuffer=tl(this.gl),this.indexBuffer=nl(this.gl),this.framebuffer=Pc(this.gl),this.textureConfig=sr(this.gl,this.textureHalfFloatExtension)}get debug(){return $().getBool("DEBUG")}dispose(){if(this.disposed)return;this.program!=null&&console.warn("Disposing a GPGPUContext that still has a bound WebGLProgram. This is probably a resource leak, delete the program with GPGPUContext.deleteProgram before disposing."),this.outputTexture!=null&&console.warn("Disposing a GPGPUContext that still has a bound output matrix texture.  This is probably a resource leak, delete the output matrix texture with GPGPUContext.deleteMatrixTexture before disposing.");const e=this.gl;D(e,()=>e.finish()),D(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,null)),D(e,()=>e.deleteFramebuffer(this.framebuffer)),D(e,()=>e.bindBuffer(e.ARRAY_BUFFER,null)),D(e,()=>e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,null)),D(e,()=>e.deleteBuffer(this.indexBuffer)),this.disposed=!0}createFloat32MatrixTexture(e,t){return this.throwIfDisposed(),sl(this.gl,e,t,this.textureConfig)}createFloat16MatrixTexture(e,t){return this.throwIfDisposed(),ol(this.gl,e,t,this.textureConfig)}createUnsignedBytesMatrixTexture(e,t){return this.throwIfDisposed(),rl(this.gl,e,t,this.textureConfig)}uploadPixelDataToTexture(e,t){this.throwIfDisposed(),ul(this.gl,e,t)}uploadDenseMatrixToTexture(e,t,s,o){this.throwIfDisposed(),ll(this.gl,e,t,s,o,this.textureConfig)}createFloat16PackedMatrixTexture(e,t){return this.throwIfDisposed(),il(this.gl,e,t,this.textureConfig)}createPackedMatrixTexture(e,t){return this.throwIfDisposed(),al(this.gl,e,t,this.textureConfig)}deleteMatrixTexture(e){this.throwIfDisposed(),this.outputTexture===e&&(ro(this.gl,this.framebuffer),this.outputTexture=null),D(this.gl,()=>this.gl.deleteTexture(e))}downloadByteEncodedFloatMatrixFromOutputTexture(e,t,s){return this.downloadMatrixDriver(e,()=>fl(this.gl,t,s,this.textureConfig))}downloadPackedMatrixFromBuffer(e,t,s,o,r,a){return pl(this.gl,e,t,s,o,r,a,this.textureConfig)}downloadFloat32MatrixFromBuffer(e,t){return hl(this.gl,e,t)}createBufferFromTexture(e,t,s){this.bindTextureToFrameBuffer(e);const o=dl(this.gl,t,s,this.textureConfig);return this.unbindTextureToFrameBuffer(),o}createAndWaitForFence(){const e=this.createFence(this.gl);return this.pollFence(e)}createFence(e){let t,s;if($().getBool("WEBGL_FENCE_API_ENABLED")){const o=e,r=o.fenceSync(o.SYNC_GPU_COMMANDS_COMPLETE,0);e.flush(),s=()=>{const a=o.clientWaitSync(r,0,0);return a===o.ALREADY_SIGNALED||a===o.CONDITION_SATISFIED},t=r}else $().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0?(t=this.beginQuery(),this.endQuery(),s=()=>this.isQueryAvailable(t,$().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))):s=()=>!0;return{query:t,isFencePassed:s}}downloadMatrixFromPackedTexture(e,t,s){return this.downloadMatrixDriver(e,()=>ml(this.gl,t,s))}createProgram(e){this.throwIfDisposed();const t=this.gl;this.vertexShader==null&&(this.vertexShader=el(t));const s=kc(t);D(t,()=>t.attachShader(s,this.vertexShader)),D(t,()=>t.attachShader(s,e)),Nc(t,s);const o=Object.assign(s,{vao:this.createVertexArray()});return this.debug&&Jn(t,o),o}buildVao(e){this.setProgram(e),this.bindVertexArray(e.vao);const t=this.gl;D(t,()=>t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,this.indexBuffer)),cl(t,e,this.vertexBuffer)}deleteProgram(e){this.throwIfDisposed(),e===this.program&&(this.program=null),e!=null&&(D(this.gl,()=>this.gl.deleteProgram(e)),this.deleteVertexArray(e.vao))}setProgram(e){this.throwIfDisposed(),this.program=e,this.program!=null&&this.debug&&Jn(this.gl,this.program),D(this.gl,()=>this.gl.useProgram(e))}getUniformLocation(e,t,s=!0){return this.throwIfDisposed(),s?Lc(this.gl,e,t):Bc(this.gl,e,t)}getAttributeLocation(e,t){return this.throwIfDisposed(),D(this.gl,()=>this.gl.getAttribLocation(e,t))}getUniformLocationNoThrow(e,t){return this.throwIfDisposed(),this.gl.getUniformLocation(e,t)}setInputMatrixTexture(e,t,s){this.throwIfDisposed(),this.throwIfNoProgram(),Mc(this.gl,e,t,s)}setOutputMatrixTexture(e,t,s){this.setOutputMatrixTextureDriver(e,s,t)}setOutputPackedMatrixTexture(e,t,s){this.throwIfDisposed();const[o,r]=tn(t,s);this.setOutputMatrixTextureDriver(e,o,r)}setOutputMatrixWriteRegion(e,t,s,o){this.setOutputMatrixWriteRegionDriver(s,e,o,t)}setOutputPackedMatrixWriteRegion(e,t,s,o){throw new Error("setOutputPackedMatrixWriteRegion not implemented.")}debugValidate(){this.program!=null&&Jn(this.gl,this.program),mn(this.gl)}executeProgram(){this.throwIfDisposed(),this.throwIfNoProgram();const e=this.gl;if(this.debug){const t=this.getVertexArray();console.assert(t===this.program.vao,"VAO changed between setProgram and executeProgram!"),this.debugValidate()}D(e,()=>e.drawElements(e.TRIANGLES,6,e.UNSIGNED_SHORT,0))}blockUntilAllProgramsCompleted(){this.throwIfDisposed(),D(this.gl,()=>this.gl.finish())}getQueryTimerExtension(){return this.disjointQueryTimerExtension==null&&(this.disjointQueryTimerExtension=pn(this.gl,$().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2?"EXT_disjoint_timer_query_webgl2":"EXT_disjoint_timer_query")),this.disjointQueryTimerExtension}getQueryTimerExtensionWebGL2(){return this.getQueryTimerExtension()}getQueryTimerExtensionWebGL1(){return this.getQueryTimerExtension()}beginQuery(){if($().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2){const s=this.gl,o=this.getQueryTimerExtensionWebGL2(),r=s.createQuery();return s.beginQuery(o.TIME_ELAPSED_EXT,r),r}const e=this.getQueryTimerExtensionWebGL1(),t=e.createQueryEXT();return e.beginQueryEXT(e.TIME_ELAPSED_EXT,t),t}endQuery(){if($().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2){const t=this.gl,s=this.getQueryTimerExtensionWebGL2();t.endQuery(s.TIME_ELAPSED_EXT);return}const e=this.getQueryTimerExtensionWebGL1();e.endQueryEXT(e.TIME_ELAPSED_EXT)}async waitForQueryAndGetTime(e){return await Us(()=>this.disposed||this.isQueryAvailable(e,$().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))),this.getQueryTime(e,$().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))}getQueryTime(e,t){if(t===0)return null;if(t===2){const s=this.gl;return s.getQueryParameter(e,s.QUERY_RESULT)/1e6}else{const s=this.getQueryTimerExtensionWebGL1();return s.getQueryObjectEXT(e,s.QUERY_RESULT_EXT)/1e6}}isQueryAvailable(e,t){if(t===0)return!0;if(t===2){const s=this.gl,o=this.getQueryTimerExtensionWebGL2(),r=s.getQueryParameter(e,s.QUERY_RESULT_AVAILABLE);return this.disjoint==null&&(this.disjoint=this.gl.getParameter(o.GPU_DISJOINT_EXT)),r&&!this.disjoint}else{const s=this.getQueryTimerExtensionWebGL1(),o=s.getQueryObjectEXT(e,s.QUERY_RESULT_AVAILABLE_EXT);return this.disjoint==null&&(this.disjoint=this.gl.getParameter(s.GPU_DISJOINT_EXT)),o&&!this.disjoint}}pollFence(e){return new Promise(t=>{this.addItemToPoll(()=>e.isFencePassed(),()=>t())})}pollItems(){const e=mx(this.itemsToPoll.map(t=>t.isDoneFn));for(let t=0;t<=e;++t){const{resolveFn:s}=this.itemsToPoll[t];s()}this.itemsToPoll=this.itemsToPoll.slice(e+1)}addItemToPoll(e,t){if(this.itemsToPoll.push({isDoneFn:e,resolveFn:t}),this.itemsToPoll.length>1)return;let s;"setTimeoutCustom"in $().platform&&(s=$().platform.setTimeoutCustom.bind($().platform)),Us(()=>(this.pollItems(),this.itemsToPoll.length===0),()=>0,null,s)}bindTextureToFrameBuffer(e){this.throwIfDisposed(),es(this.gl,e,this.framebuffer),this.debug&&mn(this.gl)}unbindTextureToFrameBuffer(){this.outputTexture!=null?(es(this.gl,this.outputTexture,this.framebuffer),this.debug&&mn(this.gl)):ro(this.gl,this.framebuffer)}downloadMatrixDriver(e,t){this.bindTextureToFrameBuffer(e);const s=t();return this.unbindTextureToFrameBuffer(),s}setOutputMatrixTextureDriver(e,t,s){this.throwIfDisposed();const o=this.gl;es(o,e,this.framebuffer),this.debug&&mn(o),this.outputTexture=e,D(o,()=>o.viewport(0,0,t,s)),D(o,()=>o.scissor(0,0,t,s))}setOutputMatrixWriteRegionDriver(e,t,s,o){this.throwIfDisposed(),D(this.gl,()=>this.gl.scissor(e,t,s,o))}throwIfDisposed(){if(this.disposed)throw new Error("Attempted to use disposed GPGPUContext.")}throwIfNoProgram(){if(this.program==null)throw new Error("No GPU program is currently set.")}}function mx(n){let e=0;for(;e<n.length&&n[e]();++e);return e-1}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ot(n,e){Array.isArray(n)||(n=[n]),n.forEach(t=>{t!=null&&R(t.dtype!=="complex64",()=>`${e} does not support complex64 tensors in the CPU backend.`)})}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gl(n){const e=new Float32Array(n.length);for(let t=0;t<n.length;++t)e[t]=Math.abs(n[t]);return e}const gx=n=>{const{x:e}=n.inputs,t=n.backend;Ot(e,"abs");let s=new Float32Array(A(e.shape));const o=t.data.get(e.dataId).values;return s=gl(o),t.makeOutput(s,e.shape,e.dtype)},RR={kernelName:go,backendName:"cpu",kernelFunc:gx};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function we(n){return(e,t,s,o,r)=>{const a=ae(e,t),i=a.length,c=J(a),l=A(a),u=Ue(r,l),d=e.length,h=t.length,f=J(e),p=J(t),x=St(e,a),g=St(t,a);if(x.length+g.length===0)for(let m=0;m<u.length;++m)u[m]=n(s[m%s.length],o[m%o.length]);else for(let m=0;m<u.length;++m){const b=Dn(m,i,c),C=b.slice(-d);x.forEach(k=>C[k]=0);const y=Ht(C,d,f),S=b.slice(-h);g.forEach(k=>S[k]=0);const N=Ht(S,h,p);u[m]=n(s[y],o[N])}return[u,a]}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Es(n){const{inputs:e,backend:t}=n,{real:s,imag:o}=e,r=t.data.get(s.dataId).values,a=t.data.get(o.dataId).values,i=t.makeTensorInfo(s.shape,"complex64"),c=t.data.get(i.dataId);return c.complexTensorInfos={real:t.makeTensorInfo(s.shape,"float32",r),imag:t.makeTensorInfo(o.shape,"float32",a)},i}const ER={kernelName:wo,backendName:"cpu",kernelFunc:Es};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function co(n,e,t="float32"){if(t==="complex64"){const o=co(n,e,"float32"),r=co(n,e,"float32");return Es({inputs:{real:o,imag:r},backend:n})}const s=We(A(e),t);return n.makeTensorInfo(e,t,s)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lo(n){const{inputs:e,backend:t}=n,{x:s}=e;return t.incRef(s.dataId),{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}const kR={kernelName:bs,backendName:"cpu",kernelFunc:lo};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xl(n){const{inputs:e,backend:t}=n,{input:s}=e,o=t.data.get(s.dataId).complexTensorInfos.real,r=t.data.get(o.dataId).values;return t.makeTensorInfo(o.shape,o.dtype,r)}const NR={kernelName:Ba,backendName:"cpu",kernelFunc:xl};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bl(n,e,t,s){if(s==="int32"){const o=Int32Array.from(n);return[e,"int32",o]}if(s==="bool"){const o=_n([0],t),[r,a]=we((i,c)=>i!==c?1:0)(e,[],n,o,"bool");return[a,"bool",r]}throw new Error(`Error in Cast: failed to cast ${t} to ${s}`)}function En(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{dtype:r}=s;if(r==="complex64"){if(o.dtype==="complex64")return lo({inputs:{x:o},backend:t});const u=co(t,o.shape,o.dtype),d=En({inputs:{x:o},backend:t,attrs:{dtype:"float32"}}),h=Es({inputs:{real:d,imag:u},backend:t});return t.disposeIntermediateTensorInfo(u),t.disposeIntermediateTensorInfo(d),h}if(o.dtype==="complex64"){const u=xl({inputs:{input:o},backend:t}),d=En({inputs:{x:u},backend:t,attrs:{dtype:r}});return t.disposeIntermediateTensorInfo(u),d}if(!po(o.dtype,r)){const u=lo({inputs:{x:o},backend:t});return{dataId:u.dataId,shape:u.shape,dtype:r}}const a=t.data.get(o.dataId).values,[i,c,l]=bl(a,o.shape,o.dtype,r);return t.makeTensorInfo(i,c,l)}const AR={kernelName:gs,backendName:"cpu",kernelFunc:En};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $e(n,e,t,s){return t==null?({inputs:o,backend:r})=>{const{a,b:i}=o,c=r;Ot([a,i],n);const l=c.data.get(a.dataId).values,u=c.data.get(i.dataId).values,d=a.dtype==="string"?Ye(l):l,h=a.dtype==="string"?Ye(u):u,f=s||a.dtype,[p,x]=e(a.shape,i.shape,d,h,f);return c.makeTensorInfo(x,f,p)}:({inputs:o,backend:r})=>{const{a,b:i}=o,c=r;if(a.dtype==="complex64"||i.dtype==="complex64"){const l=En({inputs:{x:a},backend:c,attrs:{dtype:"complex64"}}),u=c.data.get(l.dataId),d=u.complexTensorInfos.real,h=u.complexTensorInfos.imag,f=c.data.get(d.dataId).values,p=c.data.get(h.dataId).values,x=En({inputs:{x:i},backend:c,attrs:{dtype:"complex64"}}),g=c.data.get(x.dataId),m=g.complexTensorInfos.real,b=g.complexTensorInfos.imag,C=c.data.get(m.dataId).values,y=c.data.get(b.dataId).values,[S,N,k]=t(a.shape,i.shape,f,p,C,y),T=c.makeTensorInfo(k,"float32",S),I=c.makeTensorInfo(k,"float32",N),v=Es({inputs:{real:T,imag:I},backend:c});return c.disposeIntermediateTensorInfo(l),c.disposeIntermediateTensorInfo(x),c.disposeIntermediateTensorInfo(T),c.disposeIntermediateTensorInfo(I),v}else{const l=c.data.get(a.dataId).values,u=c.data.get(i.dataId).values,d=s||a.dtype,[h,f]=e(a.shape,i.shape,l,u,d);return c.makeTensorInfo(f,d,h)}}}function fr(n){return(e,t,s,o,r,a)=>{const i=ae(e,t),c=A(i),l=i.length,u=J(i),d=Ue("float32",c),h=Ue("float32",c),f=St(e,i),p=St(t,i),x=In(s,o),g=In(r,a),m=e.length,b=J(e),C=t.length,y=J(t);if(f.length+p.length===0)for(let S=0;S<d.length;S++){const N=S%x.length,k=S%g.length,T=n(x[N*2],x[N*2+1],g[k*2],g[k*2+1]);d[S]=T.real,h[S]=T.imag}else for(let S=0;S<d.length;S++){const N=Dn(S,l,u),k=N.slice(-m);f.forEach(E=>k[E]=0);const T=Ht(k,m,b),I=N.slice(-C);p.forEach(E=>I[E]=0);const v=Ht(I,C,y),w=n(x[T*2],x[T*2+1],g[v*2],g[v*2+1]);d[S]=w.real,h[S]=w.imag}return[d,h,i]}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wl=we(((n,e)=>n+e)),xx=fr(((n,e,t,s)=>({real:n+t,imag:e+s}))),bx=$e(On,wl,xx),FR={kernelName:On,backendName:"cpu",kernelFunc:bx};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wx(n,e,t,s,o){const r=A(s),a=We(o,t);for(let i=0;i<n.length;i++){const c=n[i];if(c<0)throw new Error("Input x must be non-negative!");c>=o||(r>0?a[c]+=e[i]:a[c]+=1)}return a}function Cx(n,e,t,s=!1){const o=n.shape[0],r=n.shape[1],a=ie([o,t],e.dtype);for(let i=0;i<o;i++)for(let c=0;c<r;c++){const l=n.get(i,c);if(l<0)throw new Error("Input x must be non-negative!");l>=t||(s?a.set(1,i,l):e.size>0?a.set(a.get(i,l)+e.get(i,c),i,l):a.set(a.get(i,l)+1,i,l))}return a}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cl=we(((n,e)=>n&e)),yx=$e(xo,Cl),DR={kernelName:xo,backendName:"cpu",kernelFunc:yx};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qe(n){return(e,t,s)=>{const o=te(t,e.length);for(let r=0;r<e.length;++r)o[r]=n(e[r],s);return o}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yl(n,e,t){const s=qe(e);return dt(n,s,t)}function dt(n,e,t){return({inputs:s,attrs:o,backend:r})=>{const{x:a}=s;Ot(a,n);const i=r,c=i.data.get(a.dataId).values;let l;if(a.dtype==="string"){if(!Array.isArray(c))throw new Error("String tensor's value was not an instance of Array");l=Ye(c)}else l=c;const u=t||a.dtype,d=e(l,u,o);return i.makeTensorInfo(a.shape,u,d)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $l=qe(n=>Math.ceil(n)),$x=dt(bo,$l),OR={kernelName:bo,backendName:"cpu",kernelFunc:$x};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vx(n,e,t,s){const o=te(t,A(e));if(s&&t!=="string"){let r=0;n.forEach(a=>{const i=A(a.shape);o.set(a.vals,r),r+=i})}else{let r=0;n.forEach(a=>{const i=t==="string"?Ye(a.vals):a.vals;let c=0;for(let l=0;l<a.shape[0];++l){const u=l*e[1]+r;for(let d=0;d<a.shape[1];++d)o[u+d]=i[c++]}r+=a.shape[1]})}return o}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vl=we((n,e)=>n===e?1:0),Sx=$e(Co,vl,null,"bool"),PR={kernelName:Co,backendName:"cpu",kernelFunc:Sx};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sl=qe(n=>Math.exp(n)),Ix=dt(yo,Sl,"float32"),_R={kernelName:yo,backendName:"cpu",kernelFunc:Ix};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Il=qe(n=>Math.expm1(n)),Tx=dt($o,Il),LR={kernelName:$o,backendName:"cpu",kernelFunc:Tx};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tl=qe(n=>Math.floor(n)),Rx=dt(vo,Tl),BR={kernelName:vo,backendName:"cpu",kernelFunc:Rx};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rl=we((n,e)=>Math.floor(n/e)),Ex=$e(xs,Rl,null,"int32"),MR={kernelName:xs,backendName:"cpu",kernelFunc:Ex};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kx(n,e,t,s,o,r,a,i,c){const l=ie([s,r],t);for(let u=0;u<s;u++){const d=[];let h=0;for(let f=0;f<o;f++){const p=n[u*o+f];h+=p*a[f],d.push(p)}if(h<0||h>=c/r)throw new Error(`Invalid indices: ${d} does not index into ${i}`);for(let f=0;f<r;f++)l.values[u*r+f]=e.get(...e.indexToLoc(h*r+f))}return l}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nx(n,e,t){const s=ie(t,n.dtype);for(let o=0;o<s.size;++o){const a=s.indexToLoc(o).slice(),i=a[0],c=a[2],l=e.locToIndex([i,c]);a[2]=e.values[l];const u=n.locToIndex(a);0<=u&&u<n.values.length&&(s.values[o]=n.values[u])}return s}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const El=we((n,e)=>n>e?1:0),Ax=$e(So,El,null,"bool"),VR={kernelName:So,backendName:"cpu",kernelFunc:Ax};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kl=we((n,e)=>n>=e?1:0),Fx=$e(Io,kl,null,"bool"),UR={kernelName:Io,backendName:"cpu",kernelFunc:Fx};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nl=we((n,e)=>n<e?1:0),Dx=$e(To,Nl,null,"bool"),WR={kernelName:To,backendName:"cpu",kernelFunc:Dx};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Al=we((n,e)=>n<=e?1:0),Ox=$e(Ro,Al,null,"bool"),GR={kernelName:Ro,backendName:"cpu",kernelFunc:Ox};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Px(n,e,t){const s=(e-n)/(t-1),o=We(t,"float32");o[0]=n;for(let r=1;r<o.length;r++)o[r]=o[r-1]+s;return o}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fl=qe(n=>Math.log(n)),_x=dt(Eo,Fl),zR={kernelName:Eo,backendName:"cpu",kernelFunc:_x};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lx(n,e,t,s){const o=Ue(s,A(t));for(let r=0;r<o.length;++r){const a=r*e;let i=n[a];for(let c=0;c<e;++c){const l=n[a+c];(Number.isNaN(l)||l>i)&&(i=l)}o[r]=i}return o}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Dl=we(((n,e)=>Math.max(n,e))),Bx=$e(ws,Dl),HR={kernelName:ws,backendName:"cpu",kernelFunc:Bx};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ol=we(((n,e)=>Math.min(n,e))),Mx=$e(ko,Ol),XR={kernelName:ko,backendName:"cpu",kernelFunc:Mx};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pr=we(((n,e)=>n*e)),Vx=fr(((n,e,t,s)=>({real:n*t-e*s,imag:n*s+e*t}))),Ux=$e(Cs,pr,Vx),qR={kernelName:Cs,backendName:"cpu",kernelFunc:Ux};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pl(n,e,t){const s=Nt(-1,t);return pr([],e,s,n,t)}function Wx(n){const{inputs:e,backend:t}=n,{x:s}=e;Ot(s,"neg");const o=t.data.get(s.dataId).values,[r,a]=Pl(o,s.shape,s.dtype);return t.makeTensorInfo(a,s.dtype,r)}const jR={kernelName:Oa,backendName:"cpu",kernelFunc:Wx};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _l=we(((n,e)=>n!==e?1:0)),Gx=$e(No,_l,null,"bool"),KR={kernelName:No,backendName:"cpu",kernelFunc:Gx};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ll(n,e,t,s,o){const r=e.length,a=A(e),i=J(e),c=J(o),l=Ue(t,A(o));for(let u=0;u<a;++u){const d=Dn(u,r,i),h=new Array(d.length);for(let p=0;p<h.length;p++)h[p]=d[s[p]];const f=Ht(h,r,c);l[f]=n[u]}return l}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bl(n){const{inputs:e,attrs:t,backend:s}=n,{x:o}=e,{perm:r}=t;Ot(o,"transpose");const a=o.shape.length,i=new Array(a);for(let d=0;d<i.length;d++)i[d]=o.shape[r[d]];const c=s.data.get(o.dataId).values,l=Ll(c,o.shape,o.dtype,r,i);return{dataId:s.write(l,i,o.dtype),shape:i,dtype:o.dtype}}const YR={kernelName:Ha,backendName:"cpu",kernelFunc:Bl};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ml(n,e,t,s){const[o,r]=He(n,s),a=Ge(e,"int32"),i=We(A(o),a),c=A(r);for(let l=0;l<i.length;++l){const u=l*c;let d=1;for(let h=0;h<c;++h)d*=t[u+h];i[l]=d}return{outVals:i,outShape:o,outDtype:a}}function zx(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,keepDims:a}=s;Ot(o,"prod");const i=o.shape.length,c=ce(r,o.shape),l=Re(c,i);let u=c,d=o;const h=[];l!=null&&(d=Bl({inputs:{x:o},backend:t,attrs:{perm:l}}),h.push(d),u=Ee(u.length,i));const f=t.data.get(d.dataId).values,{outVals:p,outShape:x,outDtype:g}=Ml(d.shape,d.dtype,f,u);let m=x;return a&&(m=Xe(x,c)),h.forEach(b=>t.disposeIntermediateTensorInfo(b)),t.makeTensorInfo(m,g,p)}const QR={kernelName:La,backendName:"cpu",kernelFunc:zx};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hx(n,e,t){n.forEach((s,o)=>{if(s<0||s>=t){const r=Dn(o,e.length,J(e)).join(",");throw new Error(`indices[${r}] = ${s} is not in [0, ${t})`)}})}function Xx(n,e){for(let t=0;t<n.length;++t){const s=n[t],o=t===n.length-1?e:n[t+1].length;if(s.length===0)throw new Error("Ragged splits may not be empty");if(s[0]<0)throw new Error("Ragged splits must be non-negative");if(s[s.length-1]>o)throw new Error("Ragged splits must not point past values");for(let r=1;r<s.length;++r)if(s[r-1]>s[r])throw new Error("Ragged splits must be sorted in ascending order")}}function qx(n,e,t,s){const o=[];let r=0;const a=e.length-1+t.length,i=new Array(a).fill(null).map(()=>[0]);Xx(t,s);let c=1;for(let l=0;l<e.length-1;++l){c*=e[l];const u=e[l+1];for(let d=1;d<c+1;++d)i[l].push(d*u)}for(let l=0;l<n.length;++l){let u=n[l],d=n[l]+1;for(let h=0;h<t.length;++h){const f=t[h],p=h+e.length-1;if(p>=0){const x=i[p],g=x[x.length-1]-f[u];for(let m=u;m<d;++m)i[p].push(f[m+1]+g)}u=f[u],d=f[d]}d!==u&&(o.push([u,d]),r+=d-u)}return{outSplits:i,valueSlices:o,numValues:r}}function jx(n){const e=[];for(let t=0;t<n.length;++t){const s=n[t].length,o=te("int32",s);e.push(o),n[t].forEach((r,a)=>o[a]=r)}return e}function zr(n,e){const t=n.slice(0,e);for(;t.length<e;)t.push(1);for(let s=e;s<n.length;s++)t[e-1]*=n[s];return t}function Kx(n,e,t,s,o,r){const a=zr(e,2)[1],i=zr(r,2)[1];let c=0;for(const l of t)for(let u=l[0];u<l[1];++u){for(let d=0;d<s;++d)o[c*i+d]=n[u*a+d];++c}}function Yx(n,e,t,s,o){const r=e.slice();r[0]=o;const a=te(t,A(r)),i=n.length,c=i===0?0:i/e[0];return Kx(n,e,s,c,a,r),[a,r]}function Qx(n,e,t,s,o,r,a,i){if(n.length===0)throw new Error("paramsNestedSplits must be non empty");if(e[0].length===0)throw new Error("Split tensors must not be scalars");const c=e[0][0]-1;if(Hx(r,a,c),s.length===0)throw new Error("params.rank must be nonzero");const l=s[0],{outSplits:u,valueSlices:d,numValues:h}=qx(r,a,n,l),f=jx(u),p=Yx(t,s,o,d,h);return[f,p[0],p[1]]}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hr=2147483647;function Zx(n,e,t,s,o,r,a){if(e.length>1)throw new Error("starts must be a scalar or vector");if(o.length>1)throw new Error("limits must be a scalar or vector");if(a.length>1)throw new Error("deltas must be a scalar or vector");const i=e.length===0,c=o.length===0,l=a.length===0,u=[];i||u.push(e[0]),c||u.push(o[0]),l||u.push(a[0]);for(let g=1;g<u.length;++g)if(u[g]!==u[g-1])throw new Error("starts, limits, and deltas must have the same shape");const d=u.length===0?1:u[0],h=te("int32",d+1);h[0]=0;for(let g=0;g<d;++g){const m=i?n[0]:n[g],b=c?s[0]:s[g],C=l?r[0]:r[g];if(C===0)throw new Error("Requires delta != 0");let y;if(C>0&&b<m||C<0&&b>m)y=0;else if(y=Math.ceil(Math.abs((b-m)/C)),y>Hr)throw new Error(`Requires ((limit - start) / delta) <= ${Hr}`);h[g+1]=h[g]+y}const f=h[d],p=te(t,f);let x=0;for(let g=0;g<d;++g){const m=h[g+1]-h[g];let b=i?n[0]:n[g];const C=l?r[0]:r[g];for(let y=0;y<m;++y)p[x++]=b,b+=C}return[h,p]}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var ke=Oe;class us{constructor(e,t,s,o,r,a,i,c,l,u){this.shape=e,this.shapeShape=t,this.values=s,this.valuesShape=o,this.valuesDType=r,this.defaultValue=a,this.defaultValueShape=i,this.rowPartitionValues=c,this.rowPartitionValuesShapes=l,this.rowPartitionTypes=Gi(u),this.raggedRank=zi(this.rowPartitionTypes)}getRowPartitionTypeByDimension(e){return this.rowPartitionTypes[0]===ke.FIRST_DIM_SIZE?this.rowPartitionTypes[e+1]:this.rowPartitionTypes[e]}getRowPartitionTensor(e){return this.rowPartitionTypes[0]===ke.FIRST_DIM_SIZE?this.rowPartitionValues[e+1]:this.rowPartitionValues[e]}getMaxWidth(e){const t=this.getRowPartitionTensor(e-1);switch(this.getRowPartitionTypeByDimension(e-1)){case ke.VALUE_ROWIDS:return us.getMaxWidthValueRowID(t);case ke.ROW_SPLITS:return us.getMaxWidthRowSplit(t);default:throw new Error(`Cannot handle partition type ${ke[this.getRowPartitionTypeByDimension(e-1)]}`)}}static getMaxWidthRowSplit(e){const t=e.length;if(t===0||t===1)return 0;let s=0;for(let o=0;o<t-1;++o){const r=e[o+1]-e[o];r>s&&(s=r)}return s}static getMaxWidthValueRowID(e){const t=e.length;if(t===0)return 0;let s=0,o=e[0],r=0;for(let a=1;a<t;++a){const i=e[a];i!==o&&(o=i,r=Math.max(a-s,r),s=a)}return Math.max(t-s,r)}tensorShapeFromTensor(e,t,s=!0){if(t.length===0){if(e[0]===-1)return[];throw new Error("The only valid scalar shape tensor is the fully unknown shape specified as -1.")}return qr(e,s)}calculateOutputSize(e){const t=this.valuesShape,s=this.defaultValueShape;Hi(s,t);const o=this.tensorShapeFromTensor(this.shape,this.shapeShape),a=Wi(this.raggedRank,o,t);a[0]<0&&(a[0]=e);for(let i=1;i<=this.raggedRank;++i)a[i]<0&&(a[i]=this.getMaxWidth(i));return a}calculateFirstParentOutputIndex(e,t,s){const o=Math.min(e,s),r=[];let a=0;for(let i=0;i<o;++i,a+=t)r.push(a);for(let i=o;i<e;++i)r.push(-1);return R(r.length===e,()=>"Final length of result must be equal to firstDimension."),r}calculateOutputIndexRowSplit(e,t,s,o){const r=e.length,a=[];for(let i=0;i<r-1;++i){const c=e[i+1]-e[i];let l=Math.min(o,c),u=t[i];u===-1&&(l=0);for(let d=0;d<l;++d)a.push(u),u+=s;for(let d=0;d<c-l;++d)a.push(-1)}if(r>0&&a.length!==e[r-1])throw new Error("Invalid row split size.");return a}calculateOutputIndexValueRowID(e,t,s,o){const r=e.length,a=[];if(r===0)return[];let i=0,c=e[0];if(c>=t.length)throw new Error(`Got currentValueRowId=${c}, which is not less than ${t.length}`);let l=t[c];a.push(l);for(let u=1;u<r;++u){const d=e[u];if(d===c)l>=0&&(++i,i<o?l+=s:l=-1);else{if(i=0,c=d,d>=t.length)throw new Error(`Got nextValueRowId=${d} which is not less than ${t.length}`);l=t[d]}a.push(l)}if(a.length!==e.length)throw new Error("Invalid row ids.");return a}calculateOutputIndex(e,t,s,o){const r=this.getRowPartitionTensor(e),a=this.getRowPartitionTypeByDimension(e);switch(a){case ke.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(r,t,s,o);case ke.ROW_SPLITS:if(r.length-1>t.length)throw new Error(`Row partition size is greater than output size: ${r.length-1} > ${t.length}`);return this.calculateOutputIndexRowSplit(r,t,s,o);default:throw new Error(`Unsupported partition type: ${ke[a]}`)}}getFirstDimensionSize(){const e=this.rowPartitionValues[0];if(this.rowPartitionTypes.length===0)throw new Error("No row_partition_types given.");const t=this.rowPartitionTypes[0];switch(t){case ke.FIRST_DIM_SIZE:return e[0];case ke.VALUE_ROWIDS:throw new Error("Cannot handle VALUE_ROWIDS in first dimension.");case ke.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw new Error(`Cannot handle type ${ke[t]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw new Error("Invalid first partition input. Tensor requires at least one element.");const t=this.getFirstDimensionSize(),s=this.calculateOutputSize(t),o=new Array(this.raggedRank+1);o[o.length-1]=1;for(let c=o.length-2;c>=0;--c)o[c]=o[c+1]*s[c+1];const r=qr(s,!1),a=te(this.valuesDType,A(r));if(o[0]*s[0]>0){let c=this.calculateFirstParentOutputIndex(t,o[0],s[0]);for(let l=1;l<=this.raggedRank;++l)c=this.calculateOutputIndex(l-1,c,o[l],s[l]);this.setOutput(this.raggedRank,c,a,r)}return[r,a]}setOutput(e,t,s,o){if(s.length===0)return;const r=this.values,a=s;let i=o.slice();i=i.slice(e+1);const c=A(i),l=t.length;let u=this.defaultValue;if(u.length!==c&&u.length!==1){const p=this.defaultValueShape;ee(()=>{const x=Uo(u,p);u=Lp(x,i).dataSync()})}let d=0,h=0,f=0;for(let p=0;p<=l;++p){let x=p<l?t[p]:-1;if(x===f){++f;continue}if(h<f){const g=r.subarray(d*c),m=a.subarray(h*c),b=(f-h)*c;Xr(m,g,b)}if(p>=l){const g=s.length;x=Math.floor(g/c)}if(x>f)if(this.defaultValue.length===1)a.subarray(f*c,x*c).fill(this.defaultValue[0]),f=x;else for(;x>f;){const g=a.slice(f*c);Xr(g,u,c),++f}x<0?(d=p+1,h=f):(d=p,h=f,f=h+1)}}}function Xr(n,e,t){for(let s=0;s<t;s++)n[s]=e[s]}function qr(n,e){const t=[];for(let s of n){if(s<0){if(!e)throw new Error(`Dimension ${s} must be >= 0`);if(s<-1)throw new Error(`Dimension ${s} must be >= -1`);s=-1}t.push(s)}return t}function Jx(n,e,t,s,o,r,a,i,c,l){return new us(n,e,t,s,o,r,a,i,c,l).compute()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function e0(n,e,t,s){const o=n===e,r=n<e&&t<0,a=e<n&&t>1;if(o||r||a)return We(0,s);const i=Math.abs(Math.ceil((e-n)/t)),c=We(i,s);e<n&&t===1&&(t=-1),c[0]=n;for(let l=1;l<c.length;l++)c[l]=c[l-1]+t;return c}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Vl=qe(n=>1/Math.sqrt(n)),t0=dt(Ao,Vl),ZR={kernelName:Ao,backendName:"cpu",kernelFunc:t0};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function n0(n,e,t,s,o,r,a,i,c,l){const u=[s/o,o],d=n.values,h=e.values;if(s===0)return ie(t,e.dtype);const f=c instanceof is?c:ie(u,e.dtype);typeof c=="string"||typeof c=="number"?f.values.fill(c):typeof c=="boolean"&&f.values.fill(+c);for(let p=0;p<r;p++){const x=[];let g=0;for(let m=0;m<a;m++){const b=d[p*a+m];x.push(b),g+=b*i[m]}if(g<0||g>=s/o)throw new Error(`Invalid indices: ${x} does not index into ${t}`);for(let m=0;m<o;m++)l?f.values[g*o+m]+=h[p*o+m]:f.values[g*o+m]=e.rank===0?h[0]:h[p*o+m]}return f}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const s0=qe(n=>1/(1+Math.exp(-n))),o0=yl(ys,n=>1/(1+Math.exp(-n))),JR={kernelName:ys,backendName:"cpu",kernelFunc:o0};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ul(n,e,t,s,o){const r=Yo(s,e,t),a=A(t),i=J(s);if(r){const d=Qo(e,i);return o==="string"?n.slice(d,d+a):n.subarray(d,d+a)}const c=o==="string"?Ye(n):n,l=ie(s,o,c),u=ie(t,o);for(let d=0;d<u.size;++d){const h=u.indexToLoc(d),f=h.map((p,x)=>p+e[x]);u.set(l.get(...f),...h)}return o==="string"?vc(u.values):u.values}function r0(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{begin:r,size:a}=s;Ot(o,"slice");const[i,c]=Zo(o,r,a);Ko(o,i,c);const l=t.data.get(o.dataId).values,u=Ul(l,i,c,o.shape,o.dtype);return t.makeTensorInfo(c,o.dtype,u)}const eE={kernelName:Wa,backendName:"cpu",kernelFunc:r0};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function a0(n,e,t,s,o,r,a){const i=e[0],c=r[0],l=new Array(c),u=new Array(i),d=e[1];if(c===0){if(i!==0)throw new Error(lc(i));const g=te(t,0),m=te(o,0);return[g,[0,d],m,l,u]}let h=!0,f=0;const p=new Array(c).fill(0);for(let g=0;g<i;++g){const m=n[g*d];if(m<0)throw new Error(uc(g,m));if(m>=c)throw new Error(dc(g,m,c));++p[m],h=h&&m>=f,f=m}let x=!0;for(let g=0;g<c;++g){const m=p[g]===0;l[g]=m,x=x&&!m,p[g]=Math.max(p[g],1),g>0&&(p[g]+=p[g-1])}if(x&&h){const g=n,m=s;for(let b=0;b<i;++b)u[b]=b;return[g,[i,d],m,l,u]}else{const g=p[c-1],m=te(t,g*d),b=te(o,g),C=new Array(c).fill(0);for(let y=0;y<i;++y){const S=n[y*d],N=C[S],k=(S===0?0:p[S-1])+N;C[S]++;for(let T=0;T<d;++T)m[k*d+T]=n[y*d+T];b[k]=s[y],u[y]=k}for(let y=0;y<c;++y)if(C[y]===0){const N=y===0?0:p[y-1];m[N*d+0]=y;for(let k=1;k<d;++k)m[N*d+k]=0;b[N]=a}return[m,[g,d],b,l,u]}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function i0(n,e,t,s,o){const r=A(s),a=e[0],i=o.length,c=[];let l=1,u=-1;for(let g=0;g<i;++g){const m=o[g];if(m===-1){if(u!==-1)throw new Error(hc(u,g));u=g,c.push(1)}else{if(m<0)throw new Error(fc(g,m));l*=m,c.push(m)}}if(u!==-1){if(l<=0)throw new Error(pc());const g=Math.trunc(r/l);if(l*g!==r)throw new Error(mc(s,c));c[u]=g}if(A(c)!==r)throw new Error(gc(s,c));const h=s.length,f=[];if(h>0){f[h-1]=1;for(let g=h-2;g>=0;--g)f[g]=f[g+1]*s[g+1]}const p=[];if(i>0){p[i-1]=1;for(let g=i-2;g>=0;--g)p[g]=p[g+1]*c[g+1]}const x=te(t,a*i);for(let g=0;g<a;++g){let m=0;for(let b=0;b<h;++b)m+=n[g*h+b]*f[b];for(let b=0;b<i;++b)x[g*i+b]=Math.trunc(m/p[b]),m%=p[b]}return[x,[a,i],c]}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function c0(n,e,t,s,o,r=!1,a=0){const i=s.length,c=[e[0],n.length/e[0]],l=c[1],d=i>0?o[i-1]+1:0;if(d<0)throw new Error(so());const h=e.slice();h[0]=d;const f=h.reduce((C,y)=>C*y,1),p=te(t,f);if(i===0)return d>0&&p.fill(a),[p,h];if(d<=0)throw new Error(so());let x=0,g=1,m=0,b=o[x];for(;;){let C=0;if(g<i){if(C=o[g],b===C){++g;continue}if(b>=C)throw new Error(xc())}if(b<0||b>=d)throw new Error(bc(b,d));b>m&&p.fill(a,m*l,b*l);for(let y=x;y<g;++y){const S=s[y];if(S<0||S>=c[0])throw new Error(wc(y,s[y],c[0]));for(let N=0;N<l;N++)p[b*l+N]+=n[S*l+N]}if(r)for(let y=0;y<l;y++)p[b*l+y]/=g-x;if(x=g,++g,m=b+1,b=C,g>i)break}return m<d&&p.fill(a,m*l,d*l),[p,h]}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const l0=qe(n=>Math.sqrt(n)),u0=yl($s,n=>Math.sqrt(n)),tE={kernelName:$s,backendName:"cpu",kernelFunc:u0};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wl=we(((n,e)=>{const t=n-e;return t*t})),d0=$e(Fo,Wl),nE={kernelName:Fo,backendName:"cpu",kernelFunc:d0};/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gl=qe((n,e)=>{const{pattern:t,replaceGlobal:s,rewrite:o}=e;return n.replace(new RegExp(t,s?"g":""),o)}),h0=dt(Do,Gl),sE={kernelName:Do,backendName:"cpu",kernelFunc:h0};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function f0(n,e,t,s){const o=ie(n,e.dtype);for(let r=0;r<o.size;r++){const a=o.indexToLoc(r),i=new Array(a.length);for(let c=0;c<i.length;c++)i[c]=a[c]*t[c]+s[c];o.set(e.get(...i),...a)}return o}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class p0{constructor(e,t,s,o,r,a){this.separator=ot(e),this.nGramWidths=t,this.leftPad=ot(s),this.rightPad=ot(o),this.padWidth=r,this.preserveShort=a}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,t){const s=this.getPadWidth(t);return Math.max(0,e+2*s-t+1)}createNGrams(e,t,s,o,r,a){for(let i=0;i<r;++i){const c=this.getPadWidth(a),l=Math.max(0,c-i),u=Math.max(0,c-(r-(i+1))),d=a-(l+u),h=t+(l>0?0:i-c);let f=0;f+=l*this.leftPad.length;for(let b=0;b<d;++b)f+=e[h+b].length;f+=u*this.rightPad.length;const p=l+u+d-1;f+=p*this.separator.length,s[o+i]=new Uint8Array(f);const x=s[o+i];let g=0;const m=b=>b.forEach(C=>x[g++]=C);for(let b=0;b<l;++b)m(this.leftPad),m(this.separator);for(let b=0;b<d-1;++b)m(e[h+b]),m(this.separator);if(d>0){m(e[h+d-1]);for(let b=0;b<u;++b)m(this.separator),m(this.rightPad)}else{for(let b=0;b<u-1;++b)m(this.rightPad),m(this.separator);m(this.rightPad)}}}compute(e,t){const s=e.length,o=t.length;if(o>0){let c=t[0];if(c!==0)throw new Error(`First split value must be 0, got ${c}`);for(let l=1;l<o;++l){let u=t[l]>=c;if(u=u&&t[l]<=s,!u)throw new Error(`Invalid split value ${t[l]}, must be in [${c}, ${s}]`);c=t[l]}if(c!==s)throw new Error(`Last split value must be data size. Expected ${s}, got ${c}`)}const r=o-1,a=te("int32",o);if(s===0||o===0){const c=new Array(s);for(let l=0;l<=r;++l)a[l]=0;return[c,a]}a[0]=0;for(let c=1;c<=r;++c){const l=t[c]-t[c-1];let u=0;this.nGramWidths.forEach(d=>{u+=this.getNumNGrams(l,d)}),this.preserveShort&&l>0&&u===0&&(u=1),a[c]=a[c-1]+u}const i=new Array(a[r]);for(let c=0;c<r;++c){const l=t[c];let u=a[c];if(this.nGramWidths.forEach(d=>{const h=t[c+1]-t[c],f=this.getNumNGrams(h,d);this.createNGrams(e,l,i,u,f,d),u+=f}),this.preserveShort&&u===a[c]){const d=t[c+1]-t[c];if(d===0)continue;const h=d+2*this.padWidth;this.createNGrams(e,l,i,u,1,h)}}return[i,a]}}function m0(n,e,t,s,o,r,a,i){return new p0(t,s,o,r,a,i).compute(n,e)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function g0(n,e,t,s){if(!n.length)return;if(e.length===0){for(let r=0;r<n.length;++r)s.push(n.subarray(r,r+1));return}if(e.length===1){const r=e[0];let a=n.indexOf(r);for(;a!==-1;){const i=n.subarray(0,a);(!t||i.length!==0)&&s.push(i),n=n.subarray(a+1),a=n.indexOf(r)}(!t||n.length!==0)&&s.push(n);return}let o=0;for(let r=0;r<n.length+1;r++)if(r===n.length||e.indexOf(n[r])!==-1){const a=n.subarray(o,r);(!t||a.length!==0)&&s.push(a),o=r+1}}function x0(n,e,t){const s=n.length,o=[];let r=0,a=0;const i=new Array(s);for(let h=0;h<s;++h){const f=o.length;g0(n[h],e,t,o);const p=o.length-f;i[h]=p,r+=p,a=Math.max(a,p)}const c=te("int32",r*2),l=new Array(r),u=[s,a];let d=0;for(let h=0;h<s;++h)for(let f=0;f<i[h];++f)c[d*2]=h,c[d*2+1]=f,l[d]=o[d],++d;return[c,l,u]}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function b0(n,e){const t=te("int32",n.length);for(let s=0;s<n.length;++s)t[s]=ei(n[s]).modulo(e).getLowBitsUnsigned();return t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zl=we(((n,e)=>n-e)),w0=fr(((n,e,t,s)=>({real:n-t,imag:e-s}))),C0=$e(vs,zl,w0),oE={kernelName:vs,backendName:"cpu",kernelFunc:C0};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function y0(n,e){const t=new Array(n.rank);for(let o=0;o<t.length;o++)t[o]=n.shape[o]*e[o];const s=ie(t,n.dtype);for(let o=0;o<s.values.length;++o){const r=s.indexToLoc(o),a=new Array(n.rank);for(let c=0;c<a.length;c++)a[c]=r[c]%n.shape[c];const i=n.locToIndex(a);s.values[o]=n.values[i]}return s}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xn=(n,e)=>{const t=e.value-n.value;return t===0?n.index-e.index:t};function Hl(n,e,t=0,s=n.length-1){for(;s>t;){if(s-t>600){const i=s-t+1,c=e-t+1,l=Math.log(i),u=.5*Math.exp(2*l/3),d=.5*Math.sqrt(l*u*(i-u)/i)*Math.sign(c-i/2),h=Math.max(t,Math.floor(e-c*u/i+d)),f=Math.min(s,Math.floor(e+(i-c)*u/i+d));Hl(n,e,h,f)}const o=n[e];let r=t,a=s;for(je(n,t,e),xn(n[s],o)>0&&je(n,t,s);r<a;){for(je(n,r,a),r++,a--;xn(n[r],o)<0;)r=r+1;for(;xn(n[a],o)>0;)a=a-1}xn(n[t],o)===0?je(n,t,a):(a=a+1,je(n,a,s)),a<=e&&(t=a+1),e<=a&&(s=a-1)}}function $0(n,e,t,s,o){const r=e[e.length-1],[a,i]=[n.length/r,r],c=Ue(t,a*s),l=Ue("int32",a*s);for(let d=0;d<a;d++){const h=d*i,f=n.subarray(h,h+i);let p=new Array(f.length);f.forEach((b,C)=>p[C]={value:b,index:C}),s<p.length&&(Hl(p,s),p=p.slice(0,s)),o&&p.sort(xn);const x=d*s,g=c.subarray(x,x+s),m=l.subarray(x,x+s);for(let b=0;b<s;b++)g[b]=p[b].value,m[b]=p[b].index}const u=e.slice();return u[u.length-1]=s,[ie(u,t,c),ie(u,"int32",l)]}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function v0(n,e,t,s){const o=ce(e,t)[0],r=[1,t[0],1];for(let p=0;p<o;p++)r[0]*=t[p];r[1]=t[o];for(let p=o+1;p<t.length;p++)r[2]*=t[p];const a=new Map,i=new Int32Array(t[o]),c=new is(r,s,n),l=[],u=r[0]===1&&r[2]===1;for(let p=0;p<t[o];p++){let x;if(u)x=n[p].toString();else{const m=[];for(let b=0;b<r[0];b++)for(let C=0;C<r[2];C++)m.push(c.get(b,p,C));x=m.join(",")}const g=a.get(x);if(g!=null)i[p]=g;else{const m=a.size;a.set(x,m),i[p]=m,l.push(p)}}const d=r.slice();d[1]=a.size;const h=new is(d,s);l.forEach((p,x)=>{for(let g=0;g<r[0];g++)for(let m=0;m<r[2];m++)h.set(c.get(g,p,m),g,x,m)});const f=t.slice();return f[o]=d[1],{outputValues:h.values,outputShape:f,indices:i}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const S0=Object.freeze(Object.defineProperty({__proto__:null,addImpl:wl,bincountImpl:wx,bincountReduceImpl:Cx,bitwiseAndImpl:Cl,castImpl:bl,ceilImpl:$l,concatImpl:vx,equalImpl:vl,expImpl:Sl,expm1Impl:Il,floorDivImpl:Rl,floorImpl:Tl,gatherNdImpl:kx,gatherV2Impl:Nx,greaterEqualImpl:kl,greaterImpl:El,lessEqualImpl:Al,lessImpl:Nl,linSpaceImpl:Px,logImpl:Fl,maxImpl:Lx,maximumImpl:Dl,minimumImpl:Ol,multiplyImpl:pr,negImpl:Pl,notEqualImpl:_l,prodImpl:Ml,raggedGatherImpl:Qx,raggedRangeImpl:Zx,raggedTensorToTensorImpl:Jx,rangeImpl:e0,rsqrtImpl:Vl,scatterImpl:n0,sigmoidImpl:s0,simpleAbsImpl:gl,sliceImpl:Ul,sparseFillEmptyRowsImpl:a0,sparseReshapeImpl:i0,sparseSegmentReductionImpl:c0,sqrtImpl:l0,squaredDifferenceImpl:Wl,staticRegexReplaceImpl:Gl,stridedSliceImpl:f0,stringNGramsImpl:m0,stringSplitImpl:x0,stringToHashBucketFastImpl:b0,subImpl:zl,tileImpl:y0,topKImpl:$0,transposeImpl:Ll,uniqueImpl:v0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const{addImpl:I0,bincountImpl:Xl,bincountReduceImpl:T0,bitwiseAndImpl:R0,castImpl:E0,ceilImpl:k0,concatImpl:N0,equalImpl:A0,expImpl:F0,expm1Impl:D0,floorImpl:O0,gatherNdImpl:P0,gatherV2Impl:_0,greaterImpl:L0,greaterEqualImpl:B0,lessImpl:M0,lessEqualImpl:V0,linSpaceImpl:U0,logImpl:W0,maxImpl:G0,maximumImpl:z0,minimumImpl:H0,multiplyImpl:X0,negImpl:q0,notEqualImpl:j0,prodImpl:K0,raggedGatherImpl:Y0,raggedRangeImpl:Q0,raggedTensorToTensorImpl:Z0,rangeImpl:J0,rsqrtImpl:eb,scatterImpl:tb,sigmoidImpl:nb,simpleAbsImpl:ql,sliceImpl:sb,sparseFillEmptyRowsImpl:ob,sparseReshapeImpl:rb,sparseSegmentReductionImpl:jl,sqrtImpl:ab,staticRegexReplaceImpl:ib,stridedSliceImpl:cb,stringNGramsImpl:lb,stringSplitImpl:ub,stringToHashBucketFastImpl:db,subImpl:hb,tileImpl:fb,topKImpl:pb,transposeImpl:mr,uniqueImpl:mb}=S0;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kl(n,e){return["x","y","z","w","u","v"].slice(0,e).map(t=>`${n}.${t}`)}function pe(n,e){return e===1?[n]:Kl(n,e)}function gb(n,e){if(n===1)return"rc";let t="";for(let s=0;s<n;s++)t+=e[s],s<n-1&&(t+=",");return t}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class xb{constructor(e){if(this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.enableShapeUniforms=de(this.outputShape.length),this.rank===0)this.userCode=`
        void main() {
          setOutput(vec4(getA(), 0., 0., 0.));
        }
      `;else{const t=pe("rc",this.rank),s=H(this.rank),o=this.getOutOfBoundsCondition(t),r=this.getSetup(t),a=this.getOutput(t);this.userCode=`
        void main() {
          ${s} rc = getOutputCoords();

          if(${o}) {
            setOutput(vec4(0));
          } else {
            ${r}

            setOutput(vec4(${a}));
          }
        }
      `}}getSourceCoordsArr(e){const t=[];for(let s=0;s<=1;s++)for(let o=0;o<=1;o++){let r=`${s===0?"r":"rp1"}, ${o===0?"c":"cp1"}`;for(let a=2;a<this.rank;a++)r=`${e[e.length-1-a]},`+r;t.push(r)}return t}getOutOfBoundsCondition(e){if(this.rank===1)return`rc > ${this.enableShapeUniforms?"outShape":this.outputShape[0]}`;let t="";for(let s=this.rank-2;s<this.rank;s++)t+=`${e[s]} >= ${this.enableShapeUniforms?`outShape[${s}]`:this.outputShape[s]}`,s<this.rank-1&&(t+="||");return t}getSetup(e){if(this.rank===1)return"";const t=e.slice(-2),s=this.enableShapeUniforms?`outShape[${this.rank} - 1]`:this.outputShape[this.rank-1],o=this.enableShapeUniforms?`outShape[${this.rank} - 2]`:this.outputShape[this.rank-2];return`
      int r = ${t[0]};
      int c = ${t[1]};
      int rp1 = r + 1;
      int cp1 = c + 1;

      bool cEdge = cp1 >= ${s};
      bool rEdge = rp1 >= ${o};
    `}getOutput(e){const t=this.getSourceCoordsArr(e);return this.rank===1?`getA(rc), (rc + 1 >= ${this.enableShapeUniforms?"outShape":this.outputShape[0]} ? 0. : getA(rc + 1)), 0, 0`:`getA(${t[0]}),
            cEdge ? 0. : getA(${t[1]}),
            rEdge ? 0. : getA(${t[2]}),
            rEdge || cEdge ? 0. : getA(${t[3]})`}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Yl{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"inputShape",type:"ivec3"}],this.outputShape=e,this.enableShapeUniforms=de(this.outputShape.length);let s="";for(let o=0;o<4;o++){let r="thisRC = rc;";o%2===1&&(r+="thisRC.z += 1;"),o>1&&(r+="thisRC.y += 1;"),s+=`
        ${r}
        ${o>0?"if(thisRC.y < rows && thisRC.z < cols){":""}
          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          result[${o}] =
            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
        ${o>0?"}":""}
      `}this.userCode=`
      ${bb(t,this.enableShapeUniforms)}
      ${this.enableShapeUniforms?ar():rr(e)}

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.);

        ivec3 thisRC;
        int rows = ${this.enableShapeUniforms?"outShape[1]":e[1]};
        int cols = ${this.enableShapeUniforms?"outShape[2]":e[2]};

        ${s}

        setOutput(result);
      }
    `}}function bb(n,e){return`
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${e?$g(["r","c","d"],"inputShape"):Ft(["r","c","d"],n)}
      return ivec3(r, c, d);
    }
  `}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class wb{constructor(e){this.gpgpu=e,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0,this.freeTextures={},this.usedTextures={},this.logEnabled=!1}acquireTexture(e,t,s){const o=Kr(t,s),r=Yr(e,o,s);r in this.freeTextures||(this.freeTextures[r]=[]),r in this.usedTextures||(this.usedTextures[r]=[]);const a=jr(e,o,this.gpgpu.gl,this.gpgpu.textureConfig,s);if(this.freeTextures[r].length>0){this.numFreeTextures--,this.numUsedTextures++,this._numBytesFree-=a,this.log();const c=this.freeTextures[r].pop();return this.usedTextures[r].push(c),c}let i;return o===re.PACKED_2X2_FLOAT32?i=this.gpgpu.createPackedMatrixTexture(e[0],e[1]):o===re.PACKED_2X2_FLOAT16?i=this.gpgpu.createFloat16PackedMatrixTexture(e[0],e[1]):o===re.UNPACKED_FLOAT32?i=this.gpgpu.createFloat32MatrixTexture(e[0],e[1]):o===re.UNPACKED_FLOAT16?i=this.gpgpu.createFloat16MatrixTexture(e[0],e[1]):o===re.PACKED_4X1_UNSIGNED_BYTE&&(i=this.gpgpu.createUnsignedBytesMatrixTexture(e[0],e[1])),this.usedTextures[r].push(i),this.numUsedTextures++,this._numBytesAllocated+=a,this.log(),i}releaseTexture(e,t,s,o){if(this.freeTextures==null)return;const r=Kr(s,o),a=Yr(t,r,o);a in this.freeTextures||(this.freeTextures[a]=[]);const i=jr(t,r,this.gpgpu.gl,this.gpgpu.textureConfig,o),c=$().getNumber("WEBGL_DELETE_TEXTURE_THRESHOLD");c!==-1&&this._numBytesAllocated>c?(this.gpgpu.deleteMatrixTexture(e.texture),this._numBytesAllocated-=i):(this.freeTextures[a].push(e),this.numFreeTextures++,this._numBytesFree+=i),this.numUsedTextures--;const l=this.usedTextures[a],u=l&&l.indexOf(e);if(u==null||u<0)throw new Error("Cannot release a texture that was never provided by this texture manager");l[u]=l[l.length-1],l.pop(),this.log()}log(){if(!this.logEnabled)return;const e=this.numFreeTextures+this.numUsedTextures;console.log("Free/Used",`${this.numFreeTextures} / ${this.numUsedTextures}`,`(${e})`);const t=this._numBytesFree/this._numBytesAllocated;console.log(`Bytes allocated: ${this._numBytesAllocated}`),console.log(`Bytes unused: ${this._numBytesFree} (${Math.round(100*t)}%)`)}get numBytesAllocated(){return this._numBytesAllocated}get numBytesFree(){return this._numBytesFree}getNumUsedTextures(){return this.numUsedTextures}getNumFreeTextures(){return this.numFreeTextures}dispose(){if(this.freeTextures!=null){for(const e in this.freeTextures)this.freeTextures[e].forEach(t=>{this.gpgpu.deleteMatrixTexture(t.texture)});for(const e in this.usedTextures)this.usedTextures[e].forEach(t=>{this.gpgpu.deleteMatrixTexture(t.texture)});this.freeTextures=null,this.usedTextures=null,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0}}}function Cb(n,e){const t=n;if(e===t.R32F)return 4;if(e===t.R16F)return 2;if(e===t.RGBA32F)return 16;if(e===n.RGBA)return 16;if(e===t.RGBA16F)return 8;if(e===t.RGBA8)return 4;throw new Error(`Unknown internal format ${e}`)}function jr(n,e,t,s,o){const r=yb(e,s);let a;if(o){const[c,l]=tn(n[0],n[1]);a=c*l}else{const[c,l]=Vn(n[0],n[1]);a=c*l}const i=Cb(t,r);return a*i}function yb(n,e){switch(n){case re.PACKED_2X2_FLOAT32:return dr(e);case re.PACKED_2X2_FLOAT16:return hr(e);case re.UNPACKED_FLOAT32:return cr(e);case re.UNPACKED_FLOAT16:return lr(e);case re.PACKED_4X1_UNSIGNED_BYTE:return ur(e);default:throw new Error(`Unknown physical texture type ${n}`)}}function $b(n){return $().getBool("WEBGL_RENDER_FLOAT32_ENABLED")?n?re.PACKED_2X2_FLOAT32:re.UNPACKED_FLOAT32:n?re.PACKED_2X2_FLOAT16:re.UNPACKED_FLOAT16}function Kr(n,e){if(n===ve.UPLOAD)return re.PACKED_2X2_FLOAT32;if(n===ve.RENDER||n==null)return $b(e);if(n===ve.DOWNLOAD||n===ve.PIXELS)return re.PACKED_4X1_UNSIGNED_BYTE;throw new Error(`Unknown logical texture type ${n}`)}function Yr(n,e,t){return`${n[0]}_${n[1]}_${e}_${t}`}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Be{constructor(e,t){this.variableNames=["A"],this.outputShape=e,this.enableShapeUniforms=de(this.outputShape.length),this.userCode=`
      float unaryOperation(float x) {
        ${t}
      }

      void main() {
        float x = getAAtOutCoords();
        float y = unaryOperation(x);

        setOutput(y);
      }
    `}}const Fe="if (isnan(x)) return x;",vb="return x;",Qr="return abs(x);",Sb="return (x >= 0.0) ? x : (exp(x) - 1.0);",Ib=Fe+`
  return (x < 0.0) ? 0.0 : x;
`,Tb=Fe+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,Je="return x;",Rb="return 1.0 / (1.0 + exp(-1.0 * x));";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Eb="return x;",kb=`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`,Nb=`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,Ab=`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,Fb="return 1.0 / (1.0 + exp(-1.0 * x));";class nt{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.enableShapeUniforms=de(this.outputShape.length),this.userCode=`
      vec4 unaryOperation(vec4 x) {
        ${t}
      }

      void main() {
        vec4 x = getAAtOutCoords();
        vec4 y = unaryOperation(x);

        setOutput(y);
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Db{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outputShape=e,this.enableShapeUniforms=de(this.outputShape.length);const t=e.length,s=pe("rc",t),o=H(t),r=gb(t,s),a=s.slice(-2),i=t<=1?"rc":`vec2(${a.join(",")})`;this.userCode=`
      void main() {
        ${o} rc = getOutputCoords();
        vec4 packedInput = getA(${r});

        setOutput(getChannel(packedInput, ${i}));
      }
    `}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ob=im,Pb=1e-7,_b=1e-4,Kn={};function Lb(n){return n in Kn||(Kn[n]={}),Kn[n]}const Bb=$().getNumber("CPU_HANDOFF_SIZE_THRESHOLD"),Mb=600;function Vb(){return $().global.screen==null?1024:$().global.screen.height*$().global.screen.width*window.devicePixelRatio*Mb/1024/1024}class Wn extends ga{nextDataId(){return Wn.nextDataId++}constructor(e){if(super(),this.pendingRead=new WeakMap,this.pendingDisposal=new WeakSet,this.dataRefCount=new WeakMap,this.numBytesInGPU=0,this.uploadWaitMs=0,this.downloadWaitMs=0,this.lastGlFlushTime=0,this.warnedAboutMemory=!1,this.pendingDeletes=0,this.disposed=!1,!$().getBool("HAS_WEBGL"))throw new Error("WebGL is not supported on this device");let t;if(e!=null){if(e instanceof ss)t=e;else{const s=Pe($().getNumber("WEBGL_VERSION"),e);t=new ss(s)}this.binaryCache={},this.gpgpuCreatedLocally=!1}else{const s=Pe($().getNumber("WEBGL_VERSION"));t=new ss(s),this.binaryCache=Lb($().getNumber("WEBGL_VERSION")),this.gpgpuCreatedLocally=!0}this.gpgpu=t,this.canvas=this.gpgpu.gl.canvas,this.textureManager=new wb(this.gpgpu),this.numMBBeforeWarning=Vb(),this.texData=new Au(this,Ze())}numDataIds(){return this.texData.numDataIds()-this.pendingDeletes}writeTexture(e,t,s,o,r,a){const i=this.makeTensorInfo(t,s),c=this.texData.get(i.dataId);c.isPacked=!1,c.texture={texture:e,texShape:[o,r]},c.texShape=[o,r];const l=gn(t),u=new Gr(l,!1,a),d=this.runWebGLProgram(u,[i],s,[[o,r]]);return d.shape=t,c.texture=null,this.disposeIntermediateTensorInfo(i),d.dataId}write(e,t,s){if(($().getBool("WEBGL_CHECK_NUMERICAL_PROBLEMS")||$().getBool("DEBUG"))&&this.checkNumericalProblems(e),s==="complex64"&&e!=null)throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");const o={id:this.nextDataId()};return this.texData.set(o,{shape:t,dtype:s,values:e,usage:ve.UPLOAD,refCount:1}),o}refCount(e){return this.texData.has(e)?this.texData.get(e).refCount:0}incRef(e){const t=this.texData.get(e);t.refCount++}decRef(e){if(this.texData.has(e)){const t=this.texData.get(e);t.refCount--}}move(e,t,s,o,r){if($().getBool("DEBUG")&&this.checkNumericalProblems(t),o==="complex64")throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");this.texData.set(e,{shape:s,dtype:o,values:t,usage:ve.UPLOAD,refCount:r})}disposeIntermediateTensorInfo(e){this.disposeData(e.dataId)}readSync(e){const t=this.texData.get(e),{values:s,dtype:o,complexTensorInfos:r,slice:a,shape:i,isPacked:c}=t;if(a!=null){let h;c?h=new nt(i,Je):h=new Be(i,Je);const f=this.runWebGLProgram(h,[{dataId:e,shape:i,dtype:o}],o),p=this.readSync(f.dataId);return this.disposeIntermediateTensorInfo(f),p}if(s!=null)return this.convertAndCacheOnCPU(e);if(o==="string")return s;const l=this.activeTimers!=null;let u;l&&(u=Ne());let d;if(o==="complex64"){const h=this.readSync(r.real.dataId),f=this.readSync(r.imag.dataId);d=In(h,f)}else d=this.getValuesFromTexture(e);return l&&(this.downloadWaitMs+=Ne()-u),this.convertAndCacheOnCPU(e,d)}async read(e){if(this.pendingRead.has(e)){const p=this.pendingRead.get(e);return new Promise(x=>p.push(x))}const t=this.texData.get(e),{values:s,shape:o,slice:r,dtype:a,complexTensorInfos:i,isPacked:c}=t;if(r!=null){let p;c?p=new nt(o,Je):p=new Be(o,Je);const x=this.runWebGLProgram(p,[{dataId:e,shape:o,dtype:a}],a),g=this.read(x.dataId);return this.disposeIntermediateTensorInfo(x),g}if(s!=null)return this.convertAndCacheOnCPU(e);if($().getBool("DEBUG")&&!$().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")&&$().getNumber("WEBGL_VERSION")===2)throw new Error("tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and WEBGL_VERSION=2 not yet supported.");let l=null,u;if(a!=="complex64"&&$().get("WEBGL_BUFFER_SUPPORTED")){u=this.decode(e);const p=this.texData.get(u.dataId);l=this.gpgpu.createBufferFromTexture(p.texture.texture,...qn(o))}this.pendingRead.set(e,[]),a!=="complex64"&&await this.gpgpu.createAndWaitForFence();let d;if(a==="complex64"){const p=await Promise.all([this.read(i.real.dataId),this.read(i.imag.dataId)]),x=p[0],g=p[1];d=In(x,g)}else if(l==null)d=this.getValuesFromTexture(e);else{const p=A(o);d=this.gpgpu.downloadFloat32MatrixFromBuffer(l,p)}if(u!=null&&this.disposeIntermediateTensorInfo(u),l!=null){const p=this.gpgpu.gl;D(p,()=>p.deleteBuffer(l))}const h=this.convertAndCacheOnCPU(e,d),f=this.pendingRead.get(e);return this.pendingRead.delete(e),f.forEach(p=>p(h)),this.pendingDisposal.has(e)&&(this.pendingDisposal.delete(e),this.disposeData(e)&&Ze().removeDataId(e,this),this.pendingDeletes--),h}readToGPU(e,t={}){const s=this.texData.get(e),{values:o,shape:r,slice:a,dtype:i,isPacked:c,texture:l}=s;if(i==="complex64")throw new Error("Does not support reading texture for complex64 dtype.");if(a!=null){let f;c?f=new nt(r,Je):f=new Be(r,Je);const p=this.runWebGLProgram(f,[{dataId:e,shape:r,dtype:i}],i),x=this.readToGPU(p,t);return this.disposeIntermediateTensorInfo(p),x}if(l==null)throw o!=null?new Error("Data is not on GPU but on CPU."):new Error("There is no data on GPU or CPU.");const u=this.decode(e,t.customTexShape),d=Ze().makeTensorFromTensorInfo(u),h=this.texData.get(u.dataId);return Object.assign({tensorRef:d},h.texture)}bufferSync(e){const t=this.readSync(e.dataId);if(e.dtype==="string")try{const s=t.map(o=>Ct(o));return ie(e.shape,e.dtype,s)}catch{throw new Error("Failed to decode encoded string bytes into utf-8")}return ie(e.shape,e.dtype,t)}checkNumericalProblems(e){if(e!=null)for(let t=0;t<e.length;t++){const s=e[t];if(!Ic(s))throw $().getBool("WEBGL_RENDER_FLOAT32_CAPABLE")?Error(`The value ${s} cannot be represented with your current settings. Consider enabling float32 rendering: 'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`):Error(`The value ${s} cannot be represented on this device.`)}}getValuesFromTexture(e){const{shape:t,dtype:s,isPacked:o}=this.texData.get(e),r=A(t);if($().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")){const h=this.decode(e),f=this.texData.get(h.dataId),p=this.gpgpu.downloadMatrixFromPackedTexture(f.texture.texture,...qn(t)).subarray(0,r);return this.disposeIntermediateTensorInfo(h),p}const a=$().getBool("WEBGL_PACK")&&o===!0,i=a?gn(t):t,c=a?new dx(i):new ux(i),l=this.runWebGLProgram(c,[{shape:i,dtype:s,dataId:e}],"float32"),u=this.texData.get(l.dataId),d=this.gpgpu.downloadByteEncodedFloatMatrixFromOutputTexture(u.texture.texture,u.texShape[0],u.texShape[1]).subarray(0,r);return this.disposeIntermediateTensorInfo(l),d}timerAvailable(){return $().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0}time(e){const t=this.activeTimers,s=[];let o=!1;this.programTimersStack==null?(this.programTimersStack=s,o=!0):this.activeTimers.push(s),this.activeTimers=s,e();const r=at(this.activeTimers.map(c=>c.query)).filter(c=>c!=null),a=at(this.activeTimers.map(c=>c.name)).filter(c=>c!=null);this.activeTimers=t,o&&(this.programTimersStack=null);const i={uploadWaitMs:this.uploadWaitMs,downloadWaitMs:this.downloadWaitMs,kernelMs:null,wallMs:null};return(async()=>{if($().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0){const c=await Promise.all(r);i.kernelMs=ba(c),i.getExtraProfileInfo=()=>c.map((l,u)=>({name:a[u],ms:l})).map(l=>`${l.name}: ${l.ms}`).join(", ")}else i.kernelMs={error:"WebGL query timers are not supported in this environment."};return this.uploadWaitMs=0,this.downloadWaitMs=0,i})()}memory(){return{unreliable:!1,numBytesInGPU:this.numBytesInGPU,numBytesInGPUAllocated:this.textureManager.numBytesAllocated,numBytesInGPUFree:this.textureManager.numBytesFree}}startTimer(){return $().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?this.gpgpu.beginQuery():{startMs:Ne(),endMs:null}}endTimer(e){return $().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?(this.gpgpu.endQuery(),e):(e.endMs=Ne(),e)}async getQueryTime(e){if($().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0)return this.gpgpu.waitForQueryAndGetTime(e);const t=e;return t.endMs-t.startMs}disposeData(e,t=!1){if(this.pendingDisposal.has(e))return!1;if(!this.texData.has(e))return!0;if(t?this.texData.get(e).refCount=0:this.texData.get(e).refCount--,!t&&this.texData.get(e).refCount>0)return!1;if(this.pendingRead.has(e))return this.pendingDisposal.add(e),this.pendingDeletes++,!1;this.releaseGPUData(e);const{complexTensorInfos:s}=this.texData.get(e);return s!=null&&(this.disposeData(s.real.dataId,t),this.disposeData(s.imag.dataId,t)),this.texData.delete(e),!0}releaseGPUData(e){const{texture:t,dtype:s,texShape:o,usage:r,isPacked:a,slice:i}=this.texData.get(e),c=i&&i.origDataId||e,l=this.dataRefCount.get(c);l>1?this.dataRefCount.set(c,l-1):(this.dataRefCount.delete(c),t!=null&&(this.numBytesInGPU-=this.computeBytes(o,s),this.textureManager.releaseTexture(t,o,r,a)));const u=this.texData.get(e);u.texture=null,u.texShape=null,u.isPacked=!1,u.slice=null}getTexture(e){return this.uploadToGPU(e),this.texData.get(e).texture.texture}getDataInfo(e){return this.texData.get(e)}shouldExecuteOnCPU(e,t=Bb){return $().getBool("WEBGL_CPU_FORWARD")&&e.every(s=>this.texData.get(s.dataId).texture==null&&A(s.shape)<t)}getGPGPUContext(){return this.gpgpu}where(e){Ae("tf.where() in webgl locks the UI thread. Call tf.whereAsync() instead");const t=e.dataSync();return Ob(e.shape,t)}packedUnaryOp(e,t,s){const o=new nt(e.shape,t),r=this.compileAndRun(o,[e],s);return Ze().makeTensorFromTensorInfo(r)}abs(e){if(this.shouldExecuteOnCPU([e])&&e.dtype!=="complex64"){const o=ql(this.texData.get(e.dataId).values);return this.makeOutput(e.shape,e.dtype,o)}if($().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(e,Qr,e.dtype);const t=new Be(e.shape,Qr),s=this.compileAndRun(t,[e]);return Ze().makeTensorFromTensorInfo(s)}makeTensorInfo(e,t,s){let o;if(t==="string"&&s!=null&&s.length>0&&Fn(s[0])){const r=s.map(a=>ot(a));o=this.write(r,e,t)}else o=this.write(s,e,t);return this.texData.get(o).usage=null,{dataId:o,shape:e,dtype:t}}makeOutput(e,t,s){return Ze().makeTensorFromTensorInfo(this.makeTensorInfo(e,t,s),this)}unpackTensor(e){const t=new Db(e.shape);return this.runWebGLProgram(t,[e],e.dtype)}packTensor(e){const t=new xb(e.shape);return this.runWebGLProgram(t,[e],e.dtype,null,!0)}packedReshape(e,t){const s=[Rt(e.shape),...Et(e.shape)],o={dtype:e.dtype,shape:s,dataId:e.dataId},r=[Rt(t),...Et(t)],a=new Yl(r,s),i=!0,c=[s],l=this.runWebGLProgram(a,[o],e.dtype,c,i);return{dataId:l.dataId,shape:t,dtype:l.dtype}}decode(e,t){const s=this.texData.get(e),{isPacked:o,shape:r,dtype:a}=s;if(t!=null){const h=A(r),f=t[0]*t[1]*4;R(h<=f,()=>"customTexShape is too small. Row * Column * 4 should be equal or larger than the size of the tensor data.")}const i=gn(r);let c;o?c=new lx(i):c=new cx(i);const l=!0,u=[t??qn(i)],d=this.runWebGLProgram(c,[{shape:i,dtype:a,dataId:e}],a,u,l,t);return{dtype:a,shape:r,dataId:d.dataId}}runWebGLProgram(e,t,s,o,r=!1,a){const i=this.makeTensorInfo(e.outputShape,s),c=this.texData.get(i.dataId);if(e.packedOutput&&(c.isPacked=!0),e.outPackingScheme===Tn.DENSE){const m=a??qn(e.outputShape);c.texShape=m.map(b=>b*2)}if(e.outTexUsage!=null&&(c.usage=e.outTexUsage),A(i.shape)===0)return c.values=Ue(i.dtype,0),i;const l=[],u=t.map(m=>{if(m.dtype==="complex64")throw new Error("GPGPUProgram does not support complex64 input. For complex64 dtypes, please separate the program into real and imaginary parts.");let b=this.texData.get(m.dataId);if(b.texture==null){if(!e.packedInputs&&A(m.shape)<=$().getNumber("WEBGL_SIZE_UPLOAD_UNIFORM"))return{shape:m.shape,texData:null,isUniform:!0,uniformValues:b.values};e.packedInputs&&(b.isPacked=!0,b.shape=m.shape)}if(this.uploadToGPU(m.dataId),!!b.isPacked!=!!e.packedInputs)m=b.isPacked?this.unpackTensor(m):this.packTensor(m),l.push(m),b=this.texData.get(m.dataId);else if(b.isPacked&&!Rn(b.shape,m.shape)){const C=m,y=m.shape;m.shape=b.shape,m=this.packedReshape(m,y),l.push(m),b=this.texData.get(m.dataId),C.shape=y}return{shape:m.shape,texData:b,isUniform:!1}});this.uploadToGPU(i.dataId);const d={shape:i.shape,texData:c,isUniform:!1},h=ix(e,u,d),f=this.getAndSaveBinary(h,()=>rx(this.gpgpu,e,u,d)),p=this.activeTimers!=null;let x;p&&(x=this.startTimer()),$().get("ENGINE_COMPILE_ONLY")||ax(this.gpgpu,f,u,d,o),l.forEach(m=>this.disposeIntermediateTensorInfo(m)),p&&(x=this.endTimer(x),this.activeTimers.push({name:e.constructor.name,query:this.getQueryTime(x)}));const g=$().getNumber("WEBGL_FLUSH_THRESHOLD");if(g>0){const m=Ne();m-this.lastGlFlushTime>g&&(this.gpgpu.gl.flush(),this.lastGlFlushTime=m)}if(!$().getBool("WEBGL_LAZILY_UNPACK")&&c.isPacked&&r===!1){const m=this.unpackTensor(i);return this.disposeIntermediateTensorInfo(i),m}return i}compileAndRun(e,t,s,o,r=!1){return s=s||t[0].dtype,this.runWebGLProgram(e,t,s,o,r)}getAndSaveBinary(e,t){return e in this.binaryCache||(this.binaryCache[e]=t()),this.binaryCache[e]}getTextureManager(){return this.textureManager}dispose(){this.disposed||($().getBool("IS_TEST")||Object.keys(this.binaryCache).forEach(t=>{this.gpgpu.deleteProgram(this.binaryCache[t].webGLProgram),delete this.binaryCache[t]}),this.textureManager.dispose(),this.canvas!=null&&typeof HTMLCanvasElement<"u"&&this.canvas instanceof HTMLCanvasElement?this.canvas.remove():this.canvas=null,this.gpgpuCreatedLocally&&(this.gpgpu.program=null,this.gpgpu.dispose()),this.disposed=!0)}floatPrecision(){return this.floatPrecisionValue==null&&(this.floatPrecisionValue=ee(()=>{if(!$().get("WEBGL_RENDER_FLOAT32_ENABLED")){const e=$().getBool("DEBUG");$().set("DEBUG",!1);const t=this.abs(ct(1e-8)).dataSync()[0];if($().set("DEBUG",e),t>0)return 32}return 16})),this.floatPrecisionValue}epsilon(){return this.floatPrecision()===32?Pb:_b}uploadToGPU(e){const t=this.texData.get(e),{shape:s,dtype:o,values:r,texture:a,usage:i,isPacked:c}=t;if(a!=null)return;const l=this.activeTimers!=null;let u;l&&(u=Ne());let d=t.texShape;if(d==null&&(d=Wc(s,c),t.texShape=d),r!=null){const h=gn(s);let f,p=d[1],x=d[0];const g=r instanceof Uint8Array||r instanceof Uint8ClampedArray;(c||!g)&&([p,x]=tn(d[0],d[1])),c?f=new fx(h,g):f=new Gr(h,g);const m=g?[x,p]:d,b=this.makeTensorInfo(m,o),C=this.texData.get(b.dataId);g?C.usage=ve.PIXELS:C.usage=ve.UPLOAD,C.texShape=m,this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(b.dataId),p,x,r);const y=[[x,p]],N=this.runWebGLProgram(f,[b],o,y,!0),k=this.texData.get(N.dataId);t.texShape=k.texShape,t.isPacked=k.isPacked,t.usage=k.usage,$().get("ENGINE_COMPILE_ONLY")?this.disposeData(N.dataId):(t.texture=k.texture,t.values=null,this.texData.delete(N.dataId)),this.disposeIntermediateTensorInfo(b),l&&(this.uploadWaitMs+=Ne()-u)}else{const h=this.acquireTexture(d,i,o,c);t.texture=h}}convertAndCacheOnCPU(e,t){const s=this.texData.get(e),{dtype:o}=s;return t!=null&&(s.values=Ub(t,o)),s.values}acquireTexture(e,t,s,o){if(this.numBytesInGPU+=this.computeBytes(e,s),!this.warnedAboutMemory&&this.numBytesInGPU>this.numMBBeforeWarning*1024*1024){const r=(this.numBytesInGPU/1024/1024).toFixed(2);this.warnedAboutMemory=!0,console.warn(`High memory usage in GPU: ${r} MB, most likely due to a memory leak`)}return this.textureManager.acquireTexture(e,t,o)}computeBytes(e,t){return e[0]*e[1]*yn(t)}checkCompileCompletion(){for(const[,e]of Object.entries(this.binaryCache))this.checkCompletion_(e)}async checkCompileCompletionAsync(){const e=[];if(this.gpgpu.parallelCompilationExtension){for(const[,t]of Object.entries(this.binaryCache))e.push(this.checkCompletionAsync_(t));return Promise.all(e)}else{for(const[,t]of Object.entries(this.binaryCache)){const s=new Promise(o=>{try{this.checkCompletion_(t),o(!0)}catch(r){throw r}});e.push(s)}return Promise.all(e)}}async checkCompletionAsync_(e){return this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.parallelCompilationExtension.COMPLETION_STATUS_KHR)?this.checkCompletion_(e):(await jm(),this.checkCompletionAsync_(e))}checkCompletion_(e){if(this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.gl.LINK_STATUS)===!1)throw console.log(this.gpgpu.gl.getProgramInfoLog(e.webGLProgram)),this.gpgpu.gl.getShaderParameter(e.fragmentShader,this.gpgpu.gl.COMPILE_STATUS)===!1?(or(e.source,this.gpgpu.gl.getShaderInfoLog(e.fragmentShader)),new Error("Failed to compile fragment shader.")):new Error("Failed to link vertex and fragment shaders.");return!0}getUniformLocations(){for(const e of Object.values(this.binaryCache)){this.gpgpu.buildVao(e.webGLProgram);const{variablesLocations:t,customUniformLocations:s,infLoc:o,nanLoc:r,outShapeLocation:a,outShapeStridesLocation:i,outTexShapeLocation:c}=Jc(this.gpgpu,e.program,e.webGLProgram);e.variablesLocations=t,e.customUniformLocations=s,e.infLoc=o,e.nanLoc=r,e.outShapeLocation=a,e.outShapeStridesLocation=i,e.outTexShapeLocation=c}}createTensorFromGPUData(e,t,s){e.channels=e.channels||"RGBA";const{texture:o,height:r,width:a,channels:i}=e,c=Ze().backend;if(!c.gpgpu.gl.isTexture(o))throw new Error("The texture is invalid. Also, please make sure the texture and the TFJS WebGL backend are using the same canvas. If you want to use your own custom canvas, you have to create and use the custom TFJS WebGL backend created from the canvas through 'new tf.MathBackendWebGL(customCanvas)'.");const l=c.writeTexture(o,t,s,r,a,i);return Ze().makeTensorFromDataId(l,t,s,c)}}Wn.nextDataId=0;function Ub(n,e){if(e==="float32"||e==="complex64")return n;if(e==="int32"||e==="bool"){const t=e==="int32"?new Int32Array(n.length):new Uint8Array(n.length);for(let s=0;s<t.length;++s)t[s]=Math.round(n[s]);return t}else throw new Error(`Unknown dtype ${e}`)}/** @license See the LICENSE file. */const Wb="4.22.0";/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ql(){$().set("WEBGL_FORCE_F16_TEXTURES",!0)}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Bo()&&Gf("webgl",()=>new Wn,2);const Gb={forceHalfFloat:Ql};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gr=`
  if (isnan(a)) return a;
  if (isnan(b)) return b;
`;class kt{constructor(e,t,s){this.variableNames=["A","B"],this.outputShape=ae(t,s),this.enableShapeUniforms=de(this.outputShape.length),this.userCode=`
      float binaryOperation(float a, float b) {
        ${e}
      }

      void main() {
        float a = getAAtOutCoords();
        float b = getBAtOutCoords();
        setOutput(binaryOperation(a, b));
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pt=`
  result.r = isNaN.r ? NAN : result.r;
  result.g = isNaN.g ? NAN : result.g;
  result.b = isNaN.b ? NAN : result.b;
  result.a = isNaN.a ? NAN : result.a;
`;class cn{constructor(e,t,s,o=!1){this.variableNames=["A","B"],this.supportsBroadcasting=!0,this.packedInputs=!0,this.packedOutput=!0,this.outputShape=ae(t,s);const r=this.outputShape.length;this.enableShapeUniforms=de(r);let a="";if(o)if(r===0||A(this.outputShape)===1)a=`
          result.y = 0.;
          result.z = 0.;
          result.w = 0.;
        `;else if(a=`
          ${H(r)} coords = getOutputCoords();
        `,r===1)this.enableShapeUniforms?a+=`
            result.y = (coords + 1) >= outShape ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `:a+=`
            result.y = (coords + 1) >= ${this.outputShape[0]} ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `;else{const c=pe("coords",r);this.enableShapeUniforms?a+=`
            bool nextRowOutOfBounds =
              (${c[r-2]} + 1) >= outShape[${r} - 2];
            bool nextColOutOfBounds =
              (${c[r-1]} + 1) >= outShape[${r} - 1];
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `:a+=`
            bool nextRowOutOfBounds =
              (${c[r-2]} + 1) >= ${this.outputShape[r-2]};
            bool nextColOutOfBounds =
              (${c[r-1]} + 1) >= ${this.outputShape[r-1]};
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `}this.userCode=`
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${e}
      }

      void main() {
        vec4 a = getAAtOutCoords();
        vec4 b = getBAtOutCoords();

        vec4 result = binaryOperation(a, b);
        ${a}

        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ye(n){const{inputs:e,backend:t}=n,{x:s}=e;return t.incRef(s.dataId),{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}const zb={kernelName:bs,backendName:"webgl",kernelFunc:ye};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ht(n){const{inputs:e,backend:t}=n,{real:s,imag:o}=e,r=t.makeTensorInfo(s.shape,"complex64"),a=t.texData.get(r.dataId),i=ye({inputs:{x:s},backend:t}),c=ye({inputs:{x:o},backend:t});return a.complexTensorInfos={real:i,imag:c},r}const Hb={kernelName:wo,backendName:"webgl",kernelFunc:ht};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zl="return (a < 0.) ? b * a : a;",Jl=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function Xb(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{alpha:r}=s,a=t.makeTensorInfo([],"float32",Nt(r,"float32")),i=$().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new cn(Jl,o.shape,a.shape):new kt(Zl,o.shape,a.shape),c=t.runWebGLProgram(i,[o,a],"float32");return t.disposeIntermediateTensorInfo(a),c}const qb={kernelName:Da,backendName:"webgl",kernelFunc:Xb};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const eu="return (a < 0.) ? b * a : a;",tu=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function jb(n){const{inputs:e,backend:t}=n,{x:s,alpha:o}=e,r=$().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new cn(tu,s.shape,o.shape):new kt(eu,s.shape,o.shape);return t.runWebGLProgram(r,[s,o],"float32")}const Kb={kernelName:_a,backendName:"webgl",kernelFunc:jb};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ln="if (isnan(x)) return x;";function U({opSnippet:n,packedOpSnippet:e,cpuKernelImpl:t,dtype:s}){return({inputs:o,backend:r})=>{const{x:a}=o,i=r,c=s||a.dtype;if(i.shouldExecuteOnCPU([a])&&t!=null){const d=i.texData.get(a.dataId),h=t(d.values,c);return i.makeTensorInfo(a.shape,c,h)}const l=$().getBool("WEBGL_PACK_UNARY_OPERATIONS")&&e!=null;let u;return l?u=new nt(a.shape,e):u=new Be(a.shape,n),i.runWebGLProgram(u,[a],c)}}function le({opSnippet:n,packedOpSnippet:e,checkOutOfBounds:t=!1,supportsComplex:s=!1,cpuKernelImpl:o,dtype:r}){return({inputs:a,backend:i})=>{const{a:c,b:l}=a,u=i;if(s&&c.dtype==="complex64"){const p=u.texData.get(c.dataId),x=u.texData.get(l.dataId),[g,m]=[[p.complexTensorInfos.real,x.complexTensorInfos.real],[p.complexTensorInfos.imag,x.complexTensorInfos.imag]].map(C=>{const[y,S]=C,N={dataId:y.dataId,dtype:y.dtype,shape:c.shape},k={dataId:S.dataId,dtype:S.dtype,shape:l.shape},T=new kt(n,c.shape,l.shape);return u.runWebGLProgram(T,[N,k],Ge(y.dtype,S.dtype))}),b=ht({inputs:{real:g,imag:m},backend:u});return u.disposeIntermediateTensorInfo(g),u.disposeIntermediateTensorInfo(m),b}const d=r||Ge(c.dtype,l.dtype);if((c.dtype==="string"||l.dtype==="string"||u.shouldExecuteOnCPU([c,l]))&&o!=null){const p=u.texData.get(c.dataId).values,x=u.texData.get(l.dataId).values,g=c.dtype==="string"?Ye(p):p,m=c.dtype==="string"?Ye(x):x,[b,C]=o(c.shape,l.shape,g,m,d),y=u.makeTensorInfo(C,d),S=u.texData.get(y.dataId);return S.values=b,y}const h=$().getBool("WEBGL_PACK_BINARY_OPERATIONS")&&e!=null;let f;return h?f=new cn(e,c.shape,l.shape,t):f=new kt(n,c.shape,l.shape),u.runWebGLProgram(f,[c,l],d)}}function kn(n,e=!1){if(n==="linear")return e?Eb:vb;if(n==="relu")return e?Nb:Ib;if(n==="elu")return e?kb:Sb;if(n==="relu6")return e?Ab:Tb;if(n==="prelu")return e?tu:eu;if(n==="leakyrelu")return e?Jl:Zl;if(n==="sigmoid")return e?Fb:Rb;throw new Error(`Activation ${n} has not been implemented for the WebGL backend.`)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class nu{constructor(e,t,s,o=!1,r=!1,a=!1,i=null,c=!1,l=!1){this.variableNames=["matrixA","matrixB"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=s,this.enableShapeUniforms=de(this.outputShape.length);const u=o?e[1]:e[2],d=Math.ceil(u/2),h=o?"i * 2, rc.y":"rc.y, i * 2",f=r?"rc.z, i * 2":"i * 2, rc.z",p=o?["a.xxyy","a.zzww"]:["a.xxzz","a.yyww"],x=r?["b.xzxz","b.ywyw"]:["b.xyxy","b.zwzw"];let g="",m="";i&&(c?g=`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${i}
        }`:l?g=`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${i}
        }`:g=`vec4 activation(vec4 x) {
          ${i}
        }`,m="result = activation(result);");const b=a?"result += getBiasAtOutCoords();":"";a&&this.variableNames.push("bias"),c&&this.variableNames.push("preluActivationWeights"),l&&this.variableNames.push("leakyreluAlpha");let C="rc.x",y="rc.x";e[0]<t[0]?C=`imod(rc.x, ${e[0]})`:t[0]<e[0]&&(y=`imod(rc.x, ${t[0]})`),this.userCode=`
      ${g}
      // Don't use uniform for sharedDimensionPacked for performance.
      const float sharedDimension = ${d}.0;

      vec4 dot2x2ARowBCol(ivec3 rc) {
        vec4 result = vec4(0);
        int batchA = ${C};
        int batchB = ${y};
        for (int i = 0; i < ${d}; i++) {
          vec4 a = getMatrixA(batchA, ${h});
          vec4 b = getMatrixB(batchB, ${f});

          // These swizzled products need to be separately added.
          // See: https://github.com/tensorflow/tfjs/issues/1735
          result += (${p[0]} * ${x[0]});
          result += (${p[1]} * ${x[1]});
        }
        return result;
      }

      void main() {
        ivec3 rc = getOutputCoords();
        vec4 result = dot2x2ARowBCol(rc);

        ${b}

        ${m}

        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zr={REAL:"return areal * breal - aimag * bimag;",IMAG:"return areal * bimag + aimag * breal;"};class Jr{constructor(e,t,s){this.variableNames=["AReal","AImag","BReal","BImag"],this.outputShape=ae(t,s),this.userCode=`
      float binaryOpComplex(
          float areal, float aimag, float breal, float bimag) {
        ${e}
      }

      void main() {
        float areal = getARealAtOutCoords();
        float aimag = getAImagAtOutCoords();
        float breal = getBRealAtOutCoords();
        float bimag = getBImagAtOutCoords();
        setOutput(binaryOpComplex(areal, aimag, breal, bimag));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ea="return a * b;";function xr(n){const{inputs:e,backend:t}=n,{a:s,b:o}=e,r=Ge(s.dtype,o.dtype);if(s.dtype==="complex64"){const i=t.texData.get(s.dataId),c=t.texData.get(o.dataId),l=new Jr(Zr.REAL,s.shape,o.shape),u=new Jr(Zr.IMAG,s.shape,o.shape),d=[{dataId:i.complexTensorInfos.real.dataId,dtype:i.complexTensorInfos.real.dtype,shape:s.shape},{dataId:i.complexTensorInfos.imag.dataId,dtype:i.complexTensorInfos.imag.dtype,shape:s.shape},{dataId:c.complexTensorInfos.real.dataId,dtype:c.complexTensorInfos.real.dtype,shape:o.shape},{dataId:c.complexTensorInfos.imag.dataId,dtype:c.complexTensorInfos.imag.dtype,shape:o.shape}],h=t.runWebGLProgram(l,d,"float32"),f=t.runWebGLProgram(u,d,"float32"),p=ht({inputs:{real:h,imag:f},backend:t});return t.disposeIntermediateTensorInfo(h),t.disposeIntermediateTensorInfo(f),p}if(t.shouldExecuteOnCPU([s,o])){const i=t.texData.get(s.dataId),c=t.texData.get(o.dataId),[l,u]=X0(s.shape,o.shape,i.values,c.values,r),d=t.makeTensorInfo(u,r),h=t.texData.get(d.dataId);return h.values=l,d}let a;return $().getBool("WEBGL_PACK_BINARY_OPERATIONS")?a=new cn(ea,s.shape,o.shape):a=new kt(ea,s.shape,o.shape),t.runWebGLProgram(a,[s,o],r)}const Yb={kernelName:Cs,backendName:"webgl",kernelFunc:xr};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qb(n,e,t){const s=[Rt(n.shape),...Et(n.shape)],o={dtype:n.dtype,shape:s,dataId:n.dataId},r=[Rt(e),...Et(e)],a=new Yl(r,s),i=!0,c=[s],l=t.runWebGLProgram(a,[o],n.dtype,c,i);return{dataId:l.dataId,shape:e,dtype:l.dtype}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function F(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{shape:r}=s,a=t,i=A(o.shape),c=wa(r,i),l=A(c);R(i===l,()=>`The new shape (${c}) has ${l} elements and the old shape (${o.shape}) has ${i} elements. The new shape and old shape must have the same number of elements.`);const u=a.texData.get(o.dataId);return u.isPacked&&!Rn(o.shape,c)&&!(u.texture!==null&&Rn(u.shape,c))?Qb(o,c,a):(a.incRef(o.dataId),{dataId:o.dataId,shape:c,dtype:o.dtype})}const Zb={kernelName:Va,backendName:"webgl",kernelFunc:F};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ta{constructor(e,t){this.variableNames=["x"];const{windowSize:s,batchSize:o,inSize:r,outSize:a}=e;this.outputShape=[o,a];const i=Math.floor(s/4)*4,c=s%4;let l="sumValue += dot(values, ones);";if(t!=null){const d=1/t;l=`sumValue += dot(values * ${Cn(d)?d.toPrecision(2):d}, ones);`}let u="";r%s>0&&(u=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return 0.0;
        }
      `),this.userCode=`
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${u}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${s};

        float sumValue = 0.0;

        for (int i = 0; i < ${i}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${l}
        }

        int inIdx = inOffset + ${i};
        if (${c===1}) {
          vec4 values = vec4(getValue(batch, inIdx), 0.0, 0.0, 0.0);

          ${l}
        } else if (${c===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1), 0.0, 0.0);

          ${l}
        } else if (${c===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2), 0.0);

          ${l}
        }
        setOutput(sumValue);
      }
    `}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Jb{constructor(e,t){this.variableNames=["x"];const{windowSize:s,batchSize:o,inSize:r,outSize:a}=e;this.outputShape=[o,a];let i="0.0",c="";t==="prod"?i="1.0":t==="min"?(i="1.0 / 1e-20",c="min"):t==="max"&&(i="-1.0 / 1e-20",c="max");let l=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t==="sum"?l="sumValue":t==="prod"?l="prodValue":t==="all"?l="allValue":t==="any"&&(l="anyValue");const u=Math.floor(s/4)*4,d=s%4;let h=`
      if (${t==="sum"}) {
        sumValue += dot(values, ones);
      } else if (${t==="prod"}) {
        vec2 tmp = vec2(values[0], values[1]) * vec2(values[2], values[3]);
        prodValue *= tmp[0] * tmp[1];
      } else {
        minMaxValue = ${c}(values, minMaxValue);
        if (${t==="min"} || ${t==="max"}) {
          minMaxValue = ${c}(values, minMaxValue);
          bvec4 isNaN = isnan(values);
          if (isNaN.r || isNaN.g || isNaN.b || isNaN.a) {
            minMaxValue = vec4(NAN);
          }
        }
      }
    `,f="vec4";t==="all"?(i="1.0",h=`
        bool reducedAllValue = all(values);
        float floatedReducedAllValue = float(reducedAllValue);
        allValue = float(allValue >= 1.0 && floatedReducedAllValue >= 1.0);
      `,f="bvec4"):t==="any"&&(i="0.0",h=`
        bool reducedAnyValue = any(values);
        float floatedReducedAnyValue = float(reducedAnyValue);
        anyValue = float(anyValue >= 1.0 || floatedReducedAnyValue >= 1.0);
      `,f="bvec4");let p="";r%s>0&&(p=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return initializationValue;
        }
      `),this.userCode=`
      const float initializationValue = ${i};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${p}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${s};

        vec4 minMaxValue = vec4(${i});
        float prodValue = 1.0;
        float sumValue = 0.0;
        float allValue = 1.0;
        float anyValue = 0.0;

        for (int i = 0; i < ${u}; i += 4) {
          int inIdx = inOffset + i;
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${h}
        }

        int inIdx = inOffset + ${u};
        if (${d===1}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          ${h}
        } else if (${d===2}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          ${h}
        } else if (${d===3}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          ${h}
        }
        setOutput(${l});
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ew(n){const e=[];for(;e.length===0||e[e.length-1].outSize!==1;){const t=e.length?e[e.length-1].outSize:n[1],s=Ts(t);e.push({inSize:t,windowSize:s,outSize:Math.ceil(t/s)})}return e}function _t(n,e,t,s){const o=ew(n.shape);let r=n;for(let a=0;a<o.length;a++){const{inSize:i,windowSize:c,outSize:l}=o[a];let u,d;t==="mean"?u=a===0?new ta({windowSize:c,inSize:i,batchSize:n.shape[0],outSize:l},i):new ta({windowSize:c,inSize:i,batchSize:n.shape[0],outSize:l}):u=new Jb({windowSize:c,inSize:i,batchSize:n.shape[0],outSize:l},t),d=r,r=s.runWebGLProgram(u,[r],e),d.dataId!==n.dataId&&s.disposeIntermediateTensorInfo(d)}return r}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class tw{constructor(e,t){this.variableNames=["A"];const s=new Array(e.length);for(let a=0;a<s.length;a++)s[a]=e[t[a]];this.outputShape=s,this.rank=s.length;const o=H(this.rank),r=nw(t);this.userCode=`
    void main() {
      ${o} resRC = getOutputCoords();
      setOutput(getA(${r}));
    }
    `}}function nw(n){const e=n.length;if(e>6)throw Error(`Transpose for rank ${e} is not yet supported`);const t=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u","resRC.v"],s=new Array(e);for(let o=0;o<n.length;o++)s[n[o]]=t[o];return s.join()}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class sw{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0;const s=new Array(e.length);for(let u=0;u<s.length;u++)s[u]=e[t[u]];if(this.outputShape=s,this.rank=s.length,this.rank>6)throw Error(`Packed transpose for rank ${this.rank} is not yet supported.`);const o=H(this.rank),r=Kl("rc",this.rank),a=new Array(this.rank);for(let u=0;u<t.length;u++)a[t[u]]=r[u];const i=`vec2(${a.slice(-2).join()})`,c=`++${r[this.rank-1]} < ${s[this.rank-1]}`,l=`getChannel(getA(${a.join()}), ${i})`;this.userCode=`
    void main() {
      ${o} rc = getOutputCoords();
      vec4 result = vec4(0.);
      result[0] = ${l};
      if(${c}) {
        result[1] = ${l};
      }
      --${r[this.rank-1]};
      if(++${r[this.rank-2]} < ${s[this.rank-2]}) {
        result[2] = ${l};
        if(${c}) {
          result[3] = ${l};
        }
      }
      setOutput(result);
    }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ks(n,e,t){const s=$().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new sw(n.shape,e):new tw(n.shape,e);return t.runWebGLProgram(s,[n],n.dtype)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ow(n,e,t,s){const o=e,r=n.shape.length,a=ce(o,n.shape);let i=a;const c=Re(i,r),l=c!=null;let u=n;l&&(u=ks(n,c,s),i=Ee(i.length,r)),Le("sum",i,r);const[d,h]=He(u.shape,i);let f=d;t&&(f=Xe(d,a));const p=A(h),g=A(n.shape)/p,m=F({inputs:{x:u},attrs:{shape:[g,p]},backend:s}),b=Po(n.dtype),C=_t(m,b,"sum",s),y=F({inputs:{x:C},attrs:{shape:f},backend:s});return s.disposeIntermediateTensorInfo(m),s.disposeIntermediateTensorInfo(C),l&&s.disposeIntermediateTensorInfo(u),y}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ns(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,keepDims:a}=s;return ow(o,r,a,t)}const rw={kernelName:Ga,backendName:"webgl",kernelFunc:Ns};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function me(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{perm:r}=s,a=t,i=o.shape.length,c=new Array(i);for(let u=0;u<c.length;u++)c[u]=o.shape[r[u]];let l;if(a.shouldExecuteOnCPU([o])){const d=a.texData.get(o.dataId).values,h=mr(d,o.shape,o.dtype,r,c);l=a.makeTensorInfo(c,o.dtype);const f=a.texData.get(l.dataId);f.values=h}else l=ks(o,r,a);return l}const aw={kernelName:Ha,backendName:"webgl",kernelFunc:me};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const su=1e3;function ds({a:n,b:e,transposeA:t,transposeB:s,backend:o,bias:r=null,preluActivationWeights:a=null,leakyreluAlpha:i=0,activation:c=null}){const l=n.shape.length,u=e.shape.length,d=t?n.shape[l-2]:n.shape[l-1],h=s?e.shape[u-1]:e.shape[u-2],f=t?n.shape[l-1]:n.shape[l-2],p=s?e.shape[u-2]:e.shape[u-1],x=n.shape.slice(0,-2),g=e.shape.slice(0,-2),m=A(x),b=A(g),y=ae(n.shape.slice(0,-2),e.shape.slice(0,-2)).concat([f,p]);R(d===h,()=>`Error in matMul: inner shapes (${d}) and (${h}) of Tensors with shapes ${n.shape} and ${e.shape} and transposeA=${t} and transposeB=${s} must match.`);const S=t?[m,d,f]:[m,f,d],N=s?[b,p,h]:[b,h,p],k=F({inputs:{x:n},backend:o,attrs:{shape:S}}),T=F({inputs:{x:e},backend:o,attrs:{shape:N}}),I=[k,T],v=Math.max(m,b),w=t?k.shape[1]:k.shape[2],E=r!=null,_=a!=null,B=c==="leakyrelu",L=c!=null?kn(c,!0):null,W=E||_||B||L!=null;let j;if((f===1||p===1)&&w>su&&W===!1){let K=k,X=T;t&&(K=me({inputs:{x:k},backend:o,attrs:{perm:[0,2,1]}}),I.push(K)),s&&(X=me({inputs:{x:T},backend:o,attrs:{perm:[0,2,1]}}),I.push(X));const ne=p!==1,Z=p===1;let ue=K;ne&&(ue=F({inputs:{x:K},backend:o,attrs:{shape:[v,w,1]}}),I.push(ue));const Lt=p===1?2:1;let Fs=X;Z&&(Fs=F({inputs:{x:X},backend:o,attrs:{shape:[v,1,w]}}),I.push(Fs));const Cr=xr({inputs:{a:ue,b:Fs},backend:o});j=Ns({inputs:{x:Cr},backend:o,attrs:{axis:Lt,keepDims:!0}}),I.push(Cr)}else{const K=Ge(n.dtype,e.dtype),X=new nu(S,N,[v,f,p],t,s,E,L,_,B),ne=[k,T];if(r!=null&&ne.push(r),_&&ne.push(a),B){const Z=o.makeTensorInfo([],"float32",Nt(i,"float32"));ne.push(Z),I.push(Z)}j=o.runWebGLProgram(X,ne,K)}const G=F({inputs:{x:j},backend:o,attrs:{shape:y}});I.push(j);for(const K of I)o.disposeIntermediateTensorInfo(K);return G}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function iw(n){const{inputs:e,backend:t,attrs:s}=n,{a:o,b:r,bias:a,preluActivationWeights:i}=e,{transposeA:c,transposeB:l,activation:u,leakyreluAlpha:d}=s;return ds({a:o,b:r,transposeA:c,transposeB:l,backend:t,bias:a,preluActivationWeights:i,leakyreluAlpha:d,activation:u})}const cw={kernelName:cf,backendName:"webgl",kernelFunc:iw};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const na="return abs(x);";function lw(n){const{inputs:e,backend:t}=n,{x:s}=e;if(t.shouldExecuteOnCPU([s])&&s.dtype!=="complex64"){const r=t.texData.get(s.dataId),a=ql(r.values);return t.makeTensorInfo(s.shape,s.dtype,a)}let o;return $().getBool("WEBGL_PACK_UNARY_OPERATIONS")?o=new nt(s.shape,na):o=new Be(s.shape,na),t.runWebGLProgram(o,[s],s.dtype)}const uw={kernelName:go,backendName:"webgl",kernelFunc:lw};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dw=Fe+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return acos(x);
`,hw=U({opSnippet:dw}),fw={kernelName:ju,backendName:"webgl",kernelFunc:hw};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pw=Fe+`
  if (x < 1.0) return NAN;
return log(x + sqrt(x * x - 1.0));`,mw=U({opSnippet:pw}),gw={kernelName:Ku,backendName:"webgl",kernelFunc:mw};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sa="return a + b;",xw=le({opSnippet:sa,packedOpSnippet:sa,supportsComplex:!0,cpuKernelImpl:I0}),bw={kernelName:On,backendName:"webgl",kernelFunc:xw};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ww{constructor(e,t){this.outputShape=[],this.outputShape=e,this.variableNames=t.map((r,a)=>`T${a}`);const s=[];this.variableNames.forEach(r=>{s.push(`float v${r} = get${r}AtOutCoords();`)});const o=this.variableNames.map(r=>`v${r}`).join(" + ");this.userCode=`
      void main() {
        ${s.join(`
        `)}

        float result = ${o};
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Cw{constructor(e,t){this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.variableNames=t.map((r,a)=>`T${a}`);const s=[];this.variableNames.forEach(r=>{s.push(`vec4 v${r} = get${r}AtOutCoords();`)});const o=this.variableNames.map(r=>`v${r}`).join(" + ");this.userCode=`
      void main() {
        ${s.join(`
        `)}

        vec4 result = ${o};
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function os(n){const{inputs:e,backend:t}=n,s=e;if(s.length===1)return ye({inputs:{x:s[0]},backend:t});if(s.length>$().getNumber("WEBGL_MAX_TEXTURES_IN_SHADER")){const c=Math.floor(s.length/2),l=os({inputs:s.slice(0,c),backend:t}),u=os({inputs:s.slice(c),backend:t});return os({inputs:[l,u],backend:t})}const o=s.map(c=>c.dtype).reduce((c,l)=>Ge(c,l)),r=s.map(c=>c.shape),i=$().getBool("WEBGL_PACK")?new Cw(s[0].shape,r):new ww(s[0].shape,r);return t.runWebGLProgram(i,s,o)}const yw={kernelName:Yu,backendName:"webgl",kernelFunc:os};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $w(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,keepDims:a}=s,i=o.shape.length,c=ce(r,o.shape);let l=c;const u=Re(l,i);let d=o;u!=null&&(d=me({inputs:{x:o},backend:t,attrs:{perm:u}}),l=Ee(l.length,i)),Le("all",l,i);const[h,f]=He(d.shape,l),p=A(f),x=F({inputs:{x:d},backend:t,attrs:{shape:[-1,p]}}),g=_t(x,x.dtype,"all",t);let m;if(a){const b=Xe(h,c);m=F({inputs:{x:g},backend:t,attrs:{shape:b}})}else m=F({inputs:{x:g},backend:t,attrs:{shape:h}});return t.disposeIntermediateTensorInfo(x),t.disposeIntermediateTensorInfo(g),u!=null&&t.disposeIntermediateTensorInfo(d),m}const vw={kernelName:Qu,backendName:"webgl",kernelFunc:$w};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sw(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,keepDims:a}=s,i=o.shape.length,c=ce(r,o.shape);let l=c;const u=Re(l,i);let d=o;u!=null&&(d=me({inputs:{x:o},backend:t,attrs:{perm:u}}),l=Ee(l.length,i)),Le("any",l,i);const[h,f]=He(d.shape,l),p=A(f),x=F({inputs:{x:d},backend:t,attrs:{shape:[-1,p]}}),g=_t(x,x.dtype,"any",t);let m;if(a){const b=Xe(h,c);m=F({inputs:{x:g},backend:t,attrs:{shape:b}})}else m=F({inputs:{x:g},backend:t,attrs:{shape:h}});return t.disposeIntermediateTensorInfo(x),t.disposeIntermediateTensorInfo(g),u!=null&&t.disposeIntermediateTensorInfo(d),m}const Iw={kernelName:Zu,backendName:"webgl",kernelFunc:Sw};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Tw{constructor(e,t,s){this.variableNames=["A"];const{windowSize:o,batchSize:r,outSize:a}=e;s||this.variableNames.push("bestIndicesA"),this.outputShape=[r,a];const i=t==="max"?">":"<",c=s?"inOffset + i;":"round(getBestIndicesA(batch, inOffset + i));";this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${o};

        int bestIndex = inOffset;
        float bestValue = getA(batch, bestIndex);

        for (int i = 0; i < ${o}; i++) {
          int inIdx = ${c};
          float candidate = getA(batch, inIdx);
          if (candidate ${i} bestValue) {
            bestValue = candidate;
            bestIndex = inIdx;
          }
        }
        setOutput(float(bestIndex));
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Rw{constructor(e,t,s,o){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,R(e.length>2,()=>`Packed arg${s.charAt(0).toUpperCase()+s.slice(1)} supports only inputs with rank above 2.`);const r=e[e.length-1],a=Math.ceil(r/t);this.outputShape=e.slice(0,-1),a>1&&this.outputShape.push(a),o||this.variableNames.push("bestIndicesA");const i=this.outputShape,c=i.length,l=H(c),u=pe("coords",c);let d,h;if(a===1){h=c+1;const T=H(h);d=`
        ${T} sourceLocR = ${T}(${u.join()}, 0);
        ++${u[c-1]};
        ${T} sourceLocG = ${T}(${u.join()}, 0);
        ++${u[c-2]};
        ${T} sourceLocA = ${T}(${u.join()}, 0);
        --${u[c-1]};
        ${T} sourceLocB = ${T}(${u.join()}, 0);
        --${u[c-2]};`}else h=c,d=`
        ${l} sourceLocR = coords;
        ++${u[c-1]};
        ${l} sourceLocG = coords;
        ++${u[c-2]};
        ${l} sourceLocA = coords;
        --${u[c-1]};
        ${l} sourceLocB = coords;
        --${u[c-2]};`;const f=["x","y","z","w","u","v"].slice(0,h),p="."+f[h-1],x=f.map(T=>"int "+T),g=pe("sourceLocR",h-1).concat("inIdx.r"),m=pe("sourceLocG",h-1).concat("inIdx.g"),b=pe("sourceLocB",h-1).concat("inIdx.b"),C=pe("sourceLocA",h-1).concat("inIdx.a"),y=s==="max"?"greaterThan":"lessThan",S=o?"":`
          inIdx = round(vec4(getBestIndicesAChannel(${g.join()}),
                             getBestIndicesAChannel(${m.join()}),
                             getBestIndicesAChannel(${b.join()}),
                             getBestIndicesAChannel(${C.join()})));`,N=`vec4(
            getAChannel(${g.join()}),
            hasNextCol ? getAChannel(${m.join()}) : 0.,
            hasNextRow ? getAChannel(${b.join()}) : 0.,
            hasNextRow && hasNextCol ? getAChannel(${C.join()}) : 0.)`,k=o?"":`
      float getBestIndicesAChannel(${x.join()}) {
        return getChannel(getBestIndicesA(${f.join()}),
                                          vec2(${f.slice(-2).join()}));
      }`;this.userCode=`
      float getAChannel(${x.join()}) {
        return getChannel(getA(${f.join()}),
                               vec2(${f.slice(-2).join()}));
      }
      ${k}
      void main() {
        ${l} coords = getOutputCoords();
        bool hasNextCol = ${u[c-1]} < ${i[c-1]-1};
        bool hasNextRow = ${u[c-2]} < ${i[c-2]-1};
        ${d}
        ivec4 srcIdx = ivec4(sourceLocR${p}, sourceLocG${p},
          sourceLocB${p}, sourceLocA${p}) * ${t};
        ivec4 inIdx = srcIdx;
        vec4 bestIndex = vec4(inIdx);
        vec4 bestValue = ${N};

        for (int i = 0; i < ${t}; i++) {
          inIdx = srcIdx;
          ${S}
          vec4 candidate = ${N};
          bvec4 nan = isnan(candidate);
          bvec4 replace = bvec4(
            vec4(${y}(candidate, bestValue)) * (vec4(1.0) - vec4(nan)));

          bestValue = vec4(replace.x  ? candidate.x : bestValue.x,
                           replace.y  ? candidate.y : bestValue.y,
                           replace.z  ? candidate.z : bestValue.z,
                           replace.w  ? candidate.w : bestValue.w);
          bestIndex = mix(bestIndex, vec4(inIdx), vec4(replace));
          srcIdx++;
        }
        setOutput(bestIndex);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ou(n,e,t,s=null){let o=e.shape[0],r=e.shape[1];s!=null&&(o=s.shape[0],r=s.shape[1]);const a=Ts(r),i={windowSize:a,inSize:r,batchSize:o,outSize:Math.ceil(r/a)},c=new Tw(i,t,s==null),l=[e];s!=null&&l.push(s);const u=n.runWebGLProgram(c,l,"int32");if(u.shape[1]===1)return u;const d=ou(n,e,t,u);return n.disposeIntermediateTensorInfo(u),d}function ru(n,e,t,s=null){const o=s!=null?s.shape:e.shape,r=o[o.length-1],a=Ts(r),i=new Rw(o,a,t,s==null),c=s==null?[e]:[e,s],l=n.runWebGLProgram(i,c,"int32");if(l.shape.length===e.shape.length){const u=ru(n,e,t,l);return n.disposeIntermediateTensorInfo(l),u}return l}function au(n,e,t,s){const o=[t];if(Le("arg"+s.charAt(0).toUpperCase()+s.slice(1),o,e.shape.length),!$().getBool("WEBGL_PACK_REDUCE")||e.shape.length<=2){const r=[],a=n.texData.get(e.dataId),i=a!==null&&a.isPacked;let c=e;i&&(c=n.unpackTensor(e),r.push(c));const[l,u]=He(c.shape,o),d=A(u),h=F({inputs:{x:c},backend:n,attrs:{shape:[-1,d]}});r.push(h);const f=ou(n,h,s);r.push(f);const p=F({inputs:{x:f},backend:n,attrs:{shape:l}});return r.forEach(x=>n.disposeIntermediateTensorInfo(x)),p}return ru(n,e,s)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ew(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r}=s;let a=ce(r,o.shape);const i=Re(a,o.shape.length);let c=o;const l=[];i!=null&&(c=me({inputs:{x:o},backend:t,attrs:{perm:i}}),l.push(c),a=Ee(a.length,c.shape.length)),Le("argMax",[a[0]],c.shape.length);const u=au(t,c,a[0],"max");return l.forEach(d=>t.disposeIntermediateTensorInfo(d)),u}const kw={kernelName:Ju,backendName:"webgl",kernelFunc:Ew};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nw(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r}=s;let a=ce(r,o.shape);const i=Re(a,o.shape.length);let c=o;const l=[];i!=null&&(c=me({inputs:{x:o},backend:t,attrs:{perm:i}}),l.push(c),a=Ee(a.length,c.shape.length)),Le("argMin",[a[0]],c.shape.length);const u=au(t,c,a[0],"min");return l.forEach(d=>t.disposeIntermediateTensorInfo(d)),u}const Aw={kernelName:ed,backendName:"webgl",kernelFunc:Nw};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fw=Fe+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return asin(x);
`,Dw=U({opSnippet:Fw}),Ow={kernelName:td,backendName:"webgl",kernelFunc:Dw};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pw=Fe+"return log(x + sqrt(x * x + 1.0));",_w=U({opSnippet:Pw}),Lw={kernelName:nd,backendName:"webgl",kernelFunc:_w};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bw=Fe+`
  return atan(x);
`,Mw=U({opSnippet:Bw}),Vw={kernelName:sd,backendName:"webgl",kernelFunc:Mw};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Uw=gr+`
  return atan(a, b);
`,Ww=`
  vec4 result = atan(a, b);
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Pt+`
  return result;
`,Gw=le({opSnippet:Uw,packedOpSnippet:Ww}),zw={kernelName:rd,backendName:"webgl",kernelFunc:Gw};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hw=Fe+`
  if ((x < -1.0) || (x > 1.0)) return NAN;
return (log(1.0 + x) - log(1.0 - x)) / 2.0;`,Xw=U({opSnippet:Hw}),qw={kernelName:od,backendName:"webgl",kernelFunc:Xw};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Nn{constructor(e,t,s,o=!1,r=!1){if(this.variableNames=["x"],t==="avg"&&s)throw new Error("Cannot compute positions for average pool.");const a=e.filterWidth,i=e.strideHeight,c=e.strideWidth,l=e.dilationHeight,u=e.dilationWidth,d=e.effectiveFilterHeight,h=e.effectiveFilterWidth,f=e.padInfo.top,p=e.padInfo.left;this.outputShape=e.outShape;const x=t==="avg",g=`((batch  * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + d`,m=`(xR * ${e.inWidth} + xC) * ${e.inChannels} + d`;let b="0.0";if(x||(b="-1.0 / 1e-20"),s){this.userCode=`
        const ivec2 strides = ivec2(${i}, ${c});
        const ivec2 pads = ivec2(${f}, ${p});

        void main() {
          ivec4 coords = getOutputCoords();
          int batch = coords[0];
          int d = coords[3];

          ivec2 xRCCorner = coords.yz * strides - pads;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          // max/min x(?, ?, d) to get y(yR, yC, d).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;
          float avgValue = 0.0;

          for (int wR = 0; wR < ${d};
              wR += ${l}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${h};
                wC += ${u}) {
              int xC = xCCorner + wC;

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float value = getX(batch, xR, xC, d);

              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value >= currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                minMaxPosition = ${o?r?g:m:`wR * ${h} + wC`};
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}const C="max";let y=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t==="avg"&&(y="avgValue / max(count, 1.0)");const S=Math.floor(a/4)*4,N=a%4,k=`
      if (${x}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${C}(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec2 strides = ivec2(${i}, ${c});
      const ivec2 pads = ivec2(${f}, ${p});
      const float initializationValue = ${b};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xR, int xC, int d) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xR, xC, d);
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        vec4 minMaxValue = vec4(${b});
        float avgValue = 0.0;
        count = 0.0;

        for (int wR = 0; wR < ${d};
            wR += ${l}) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${S}; wC += 4) {
            int xC = xCCorner + wC * ${u};

            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              getValue(batch, xR, xC + 2 * ${u}, d),
              getValue(batch, xR, xC + 3 * ${u}, d)
            );

            ${k}
          }

          int xC = xCCorner + ${S};
          if (${N===1}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              initializationValue,
              initializationValue,
              initializationValue
            );

            ${k}
          } else if (${N===2}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              initializationValue,
              initializationValue
            );

            ${k}
          } else if (${N===3}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              getValue(batch, xR, xC + 2 * ${u}, d),
              initializationValue
            );

            ${k}
          }
        }
        setOutput(${y});
      }
    `}}class br{constructor(e,t,s,o=!1,r=!1){if(this.variableNames=["x"],t==="avg"&&s)throw new Error("Cannot compute positions for average pool.");const a=e.filterWidth,i=e.strideDepth,c=e.strideHeight,l=e.strideWidth,u=e.dilationDepth,d=e.dilationHeight,h=e.dilationWidth,f=e.effectiveFilterDepth,p=e.effectiveFilterHeight,x=e.effectiveFilterWidth,g=e.padInfo.front,m=e.padInfo.top,b=e.padInfo.left;this.outputShape=e.outShape;const C=t==="avg";let y="0.0";if(C||(y="-1.0 / 1e-20"),s){this.userCode=`
        const ivec3 strides =
            ivec3(${i}, ${c}, ${l});
        const ivec3 pads = ivec3(${g}, ${m}, ${b});

        void main() {
          ivec5 coords = getOutputCoords();
          int batch = coords.x;
          int ch = coords.u;

          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
          int xDCorner = xCorner.x;
          int xRCorner = xCorner.y;
          int xCCorner = xCorner.z;

          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;

          for (int wD = 0; wD < ${f};
              wD += ${u}) {
            int xD = xDCorner + wD;

            if (xD < 0 || xD >= ${e.inDepth}) {
              continue;
            }

            for (int wR = 0; wR < ${p};
                wR += ${d}) {
              int xR = xRCorner + wR;

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int wC = 0; wC < ${x};
                  wC += ${h}) {
                int xC = xCCorner + wC;

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float value = getX(batch, xD, xR, xC, ch);

                // If a min / max value has already been found, use it. If not,
                // use the current value.
                float currMinMaxValue = mix(
                    value, minMaxValue, minMaxValueFound);
                if (value >= currMinMaxValue) {
                  minMaxValue = value;
                  minMaxValueFound = 1.0;
                  minMaxPosition = ${o?r?`(((batch * ${e.inDepth} + xD) * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`((xD * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`wD * ${p} * ${x} +
                      wR * ${x} + wC`};
                }
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}const S="max";let N=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t==="avg"&&(N="avgValue / max(count, 1.0)");const k=Math.floor(a/4)*4,T=a%4,I=`
      if (${C}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${S}(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec3 strides =
        ivec3(${i}, ${c}, ${l});
      const ivec3 pads = ivec3(${g}, ${m}, ${b});
      const float initializationValue = ${y};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xD, int xR, int xC, int ch) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xD, xR, xC, ch);
      }

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xDCorner = xCorner.x;
        int xRCorner = xCorner.y;
        int xCCorner = xCorner.z;

        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).
        // ? = to be determined
        vec4 minMaxValue = vec4(${y});
        float avgValue = 0.0;
        count = 0.0;

        for (int wD = 0; wD < ${f};
            wD += ${u}) {
          int xD = xDCorner + wD;

          if (xD < 0 || xD >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${p};
            wR += ${d}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${k}; wC += 4) {
              int xC = xCCorner + wC * ${h};

              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${h}, ch),
                getValue(batch, xD, xR, xC + 2 * ${h}, ch),
                getValue(batch, xD, xR, xC + 3 * ${h}, ch)
              );

              ${I}
            }

            int xC = xCCorner + ${k};
            if (${T===1}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                initializationValue,
                initializationValue,
                initializationValue
              );

              ${I}
            } else if (${T===2}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${h}, ch),
                initializationValue,
                initializationValue
              );

              ${I}
            } else if (${T===3}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${h}, ch),
                getValue(batch, xD, xR, xC + 2 * ${h}, ch),
                initializationValue
              );

              ${I}
            }
          }
        }
        setOutput(${N});
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jw(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e;nn(o,"avgPool");const{filterSize:r,strides:a,pad:i,dimRoundingMode:c}=s,l=1;R(Jt(a,l),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${a} and dilations '${l}'`);const u=Zt(o.shape,r,a,l,i,c);if(u.filterWidth===1&&u.filterHeight===1&&oe(u.inShape,u.outShape))return ye({inputs:{x:o},backend:t});const d=new Nn(u,"avg",!1);return t.runWebGLProgram(d,[o],"float32")}const Kw={kernelName:ad,backendName:"webgl",kernelFunc:jw};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yw(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{filterSize:r,strides:a,pad:i,dimRoundingMode:c,dataFormat:l}=s,u=[1,1,1],d=Ln(o.shape,r,a,u,i,c,l),h=new br(d,"avg",!1);return t.runWebGLProgram(h,[o],"float32")}const Qw={kernelName:cd,backendName:"webgl",kernelFunc:Yw};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Zw{constructor(e){this.variableNames=["dy"],this.outputShape=e.inShape;const t=e.filterHeight,s=e.filterWidth,o=e.strideHeight,r=e.strideWidth,a=e.dilationHeight,i=e.dilationWidth,c=e.effectiveFilterHeight,l=e.effectiveFilterWidth,u=c-1-e.padInfo.top,d=l-1-e.padInfo.left,h=1/(t*s);this.userCode=`
      const ivec2 pads = ivec2(${u}, ${d});
      const float avgMultiplier = float(${h});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${c};
            wR += ${a}) {
          float dyR = float(dyRCorner + wR) / ${o}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${l};
            wC+= ${i}) {
            float dyC = float(dyCCorner + wC) / ${r}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);

            dotProd += dyValue * avgMultiplier;
          }
        }
        setOutput(dotProd);
      }
    `}}class Jw{constructor(e){this.variableNames=["dy"],this.outputShape=e.inShape;const t=e.filterDepth,s=e.filterHeight,o=e.filterWidth,r=e.strideDepth,a=e.strideHeight,i=e.strideWidth,c=e.dilationDepth,l=e.dilationHeight,u=e.dilationWidth,d=e.effectiveFilterDepth,h=e.effectiveFilterHeight,f=e.effectiveFilterWidth,p=d-1-e.padInfo.front,x=h-1-e.padInfo.top,g=f-1-e.padInfo.left,m=1/(t*s*o);this.userCode=`
      const ivec3 pads = ivec3(${p}, ${x}, ${g});
      const float avgMultiplier = float(${m});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${d};
            wD += ${c}) {
          float dyD = float(dyDCorner + wD) / ${r}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${h};
              wR += ${l}) {
            float dyR = float(dyRCorner + wR) / ${a}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${f};
                wC += ${u}) {
              float dyC = float(dyCCorner + wC) / ${i}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);

              dotProd += dyValue * avgMultiplier;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eC(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,input:r}=e,a=r,{filterSize:i,strides:c,pad:l,dimRoundingMode:u}=s,d=[1,1,1],h=Ln(a.shape,i,c,d,l,u),f=new Jw(h);return t.runWebGLProgram(f,[o],a.dtype)}const tC={kernelName:ld,backendName:"webgl",kernelFunc:eC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nC(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,input:r}=e,a=r;nn([o,r],"avgPoolGrad");const{filterSize:i,strides:c,pad:l}=s,u=Zt(a.shape,i,c,1,l),d=new Zw(u);return t.runWebGLProgram(d,[o],a.dtype)}const sC={kernelName:id,backendName:"webgl",kernelFunc:nC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oC(n){const{inputs:e,backend:t,attrs:s}=n,{a:o,b:r}=e,{transposeA:a,transposeB:i}=s;return ds({a:o,b:r,transposeA:a,transposeB:i,backend:t})}const rC={kernelName:ud,backendName:"webgl",kernelFunc:oC};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class aC{constructor(e,t,s,o,r,a){this.outputShape=[],this.variableNames=["x","mean","variance"],ae(e,t),ae(e,s);let i="0.0";o!=null&&(ae(e,o),this.variableNames.push("offset"),i="getOffsetAtOutCoords()");let c="1.0";r!=null&&(ae(e,r),this.variableNames.push("scale"),c="getScaleAtOutCoords()"),this.outputShape=e,this.userCode=`
      void main() {
        float x = getXAtOutCoords();
        float mean = getMeanAtOutCoords();
        float variance = getVarianceAtOutCoords();
        float offset = ${i};
        float scale = ${c};
        float inv = scale * inversesqrt(variance + float(${a}));
        setOutput(dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class iC{constructor(e,t,s,o,r,a){this.packedInputs=!0,this.packedOutput=!0,this.variableNames=["x","mean","variance"],ae(e,t),ae(e,s);let i="vec4(0.0)";o!=null&&(ae(e,o),this.variableNames.push("offset"),i="getOffsetAtOutCoords()");let c="vec4(1.0)";r!=null&&(ae(e,r),this.variableNames.push("scale"),c="getScaleAtOutCoords()"),this.outputShape=e,this.userCode=`
      void main() {
        vec4 offset = ${i};
        vec4 scale = ${c};

        vec4 x = getXAtOutCoords();
        vec4 mean = getMeanAtOutCoords();
        vec4 variance = getVarianceAtOutCoords();

        vec4 inv = scale * inversesqrt(variance + vec4(${a}));

        setOutput((x - mean) * inv + offset);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cC=({inputs:n,backend:e,attrs:t})=>{const{x:s,mean:o,variance:r,offset:a,scale:i}=n;R(o.shape.length===r.shape.length,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),R(a==null||o.shape.length===a.shape.length,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),R(i==null||o.shape.length===i.shape.length,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");let{varianceEpsilon:c}=t;c==null&&(c=.001);const l=[s,o,r];let u=null;a!=null&&(u=a.shape,l.push(a));let d=null;i!=null&&(d=i.shape,l.push(i));const h=$().getBool("WEBGL_PACK_NORMALIZATION")?new iC(s.shape,o.shape,r.shape,u,d,c):new aC(s.shape,o.shape,r.shape,u,d,c);return e.runWebGLProgram(h,l,l[0].dtype)},lC={kernelName:Vd,backendName:"webgl",kernelFunc:cC};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class uC{constructor(e){this.variableNames=["source"],this.outputShape=e,this.rank=e.length;const t=H(this.rank);this.customUniforms=[{name:"start",arrayIndex:this.rank,type:"int"}];const s=dC(this.rank);let o;const r=e.map((a,i)=>`sourceLoc.${uo[i]} = start[${i}] + coords.${uo[i]};`);o=`
        ${t} sourceLoc;
        ${t} coords = getOutputCoords();
        ${r.join(`
`)}
      `,this.userCode=`
      void main() {
        ${o}
        setOutput(getSource(${s}));
      }
    `}}const uo=["x","y","z","w","u","v"];function dC(n){if(n===1)return"sourceLoc";if(n<=6)return uo.slice(0,n).map(e=>"sourceLoc."+e).join(",");throw Error(`Slicing for rank ${n} is not yet supported`)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class hC{constructor(e){this.variableNames=["source"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.customUniforms=[{name:"start",arrayIndex:this.rank,type:"int"}];const t=H(this.rank),s=pe("coords",this.rank),o=pe("sourceLoc",this.rank),r=this.rank===1?"sourceLoc":`vec2(${o.slice(-2).join()})`,a=`getChannel(getSource(${o.join()}), ${r})`,i=`
      result.x = ${a};
      if (++${s[this.rank-1]} < ${e[this.rank-1]}) {
        ++${o[this.rank-1]};
        result.y = ${a};
        --${o[this.rank-1]};
      }
    `,c=this.rank===1?"":`
      --${s[this.rank-1]};
      if (++${s[this.rank-2]} < ${e[this.rank-2]}) {
        ++${o[this.rank-2]};
        result.z = ${a};
        if (++${s[this.rank-1]} < ${e[this.rank-1]}) {
          ++${o[this.rank-1]};
          result.w = ${a};
        }
      }
    `,l=this.rank<=4?`sourceLoc = coords +
            ${t}(${e.map((u,d)=>`start[${d}]`).join()});`:e.map((u,d)=>`${o[d]} = ${s[d]} + start[${d}];`).join(`
`);this.userCode=`
      void main() {
        ${t} coords = getOutputCoords();
        ${t} sourceLoc;
        ${l}
        vec4 result = vec4(0.);
        ${i}
        ${c}
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fC(n,e,t,s){const o=s.texData.get(n.dataId),r=s.makeTensorInfo(t,n.dtype),a=s.texData.get(r.dataId);Object.assign(a,o),a.refCount=1,a.shape=t,a.dtype=n.dtype;let i=Qo(e,J(n.shape));o.slice&&(i+=o.slice.flatOffset),a.slice={flatOffset:i,origDataId:o.slice&&o.slice.origDataId||n.dataId};const c=s.dataRefCount.get(a.slice.origDataId)||1;return s.dataRefCount.set(a.slice.origDataId,c+1),r}function un(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{begin:r,size:a}=s,[i,c]=Zo(o,r,a);if(Ko(o,i,c),A(c)===0)return t.makeTensorInfo(c,o.dtype,[]);if(t.shouldExecuteOnCPU([o])||o.dtype==="string"){const d=t.texData.get(o.dataId),h=sb(d.values,i,c,o.shape,o.dtype);return t.makeTensorInfo(c,o.dtype,h)}const{isPacked:l}=t.texData.get(o.dataId),u=Yo(o.shape,i,c);if(l||!u){const d=$().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new hC(c):new uC(c),h=[i];return t.runWebGLProgram(d,[o],o.dtype,h)}return t.uploadToGPU(o.dataId),fC(o,i,c,t)}const pC={kernelName:Wa,backendName:"webgl",kernelFunc:un};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mC=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{blockShape:r,crops:a}=s;R(o.shape.length<=4,()=>"batchToSpaceND for rank > 4 with a WebGL backend not implemented yet");const i=r.reduce((b,C)=>b*C),c=er(o.shape,r,i),l=tr(c.length,r.length),u=nr(o.shape,r,i),d=qi(a,r.length),h=ji(u,a,r.length),f=[],p=F({inputs:{x:o},backend:t,attrs:{shape:c}}),x=me({inputs:{x:p},backend:t,attrs:{perm:l}}),g=F({inputs:{x},backend:t,attrs:{shape:u}}),m=un({inputs:{x:g},backend:t,attrs:{begin:d,size:h}});return f.push(p),f.push(x),f.push(g),f.forEach(b=>t.disposeIntermediateTensorInfo(b)),m},gC={kernelName:dd,backendName:"webgl",kernelFunc:mC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xC(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,weights:r}=e,{size:a}=s,i=t.readSync(o.dataId),c=t.readSync(r.dataId),l=Xl(i,c,r.dtype,r.shape,a);return t.makeTensorInfo([a],r.dtype,l)}const bC={kernelName:hd,backendName:"webgl",kernelFunc:xC};/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wC=`
  int r = int(a.r) & int(b.r);
  int g = int(a.g) & int(b.g);
  int rb = int(a.b) & int(b.b);
  int ra = int(a.a) & int(b.a);
  return vec4(r, g, rb, ra);
`,CC=`
  return float(int(a.r) & int(b.r));
`;function yC(n){const{inputs:e,backend:t}=n,{a:s,b:o}=e,r=$().getBool("WEBGL_PACK_BINARY_OPERATIONS"),a=$().getNumber("WEBGL_VERSION");if(t.shouldExecuteOnCPU([s,o])||a===1){const c=t.texData.get(s.dataId).values,l=t.texData.get(o.dataId).values,[u,d]=R0(s.shape,o.shape,c,l,s.dtype),h=t.makeTensorInfo(d,s.dtype),f=t.texData.get(h.dataId);return f.values=u,h}let i;return r?i=new cn(wC,s.shape,o.shape,!1):i=new kt(CC,s.shape,o.shape),t.runWebGLProgram(i,[s,o],s.dtype)}const $C={kernelName:xo,backendName:"webgl",kernelFunc:yC};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vC(n){const{inputs:e,backend:t}=n,{s0:s,s1:o}=e,r=t.readSync(s.dataId),a=t.readSync(o.dataId),i=ae(Array.from(r),Array.from(a));return t.makeTensorInfo([i.length],"int32",Int32Array.from(i))}const SC={kernelName:fd,backendName:"webgl",kernelFunc:vC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const IC="return float(a != b);",iu=le({opSnippet:IC,cpuKernelImpl:j0,dtype:"bool"}),TC={kernelName:No,backendName:"webgl",kernelFunc:iu};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gn(n){const{inputs:e,backend:t}=n,{input:s}=e,o=t.texData.get(s.dataId);return ye({inputs:{x:o.complexTensorInfos.real},backend:t})}const RC={kernelName:Ba,backendName:"webgl",kernelFunc:Gn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const EC="return float(int(x));";function kC(n,e){const t=new Be(n.shape,EC),s=e.runWebGLProgram(t,[n],"int32");return{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ho(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{dtype:r}=s;if(r==="complex64"){if(o.dtype==="complex64")return ye({inputs:{x:o},backend:t});const a=eo(o.shape),i=ho({inputs:{x:o},backend:t,attrs:{dtype:"float32"}}),c=ht({inputs:{real:i,imag:a},backend:t});return a.dispose(),t.disposeIntermediateTensorInfo(i),c}if(o.dtype==="complex64"){const a=Gn({inputs:{input:o},backend:t}),i=ho({inputs:{x:a},backend:t,attrs:{dtype:r}});return t.disposeIntermediateTensorInfo(a),i}if(!po(o.dtype,r)){const a=ye({inputs:{x:o},backend:t});return{dataId:a.dataId,shape:a.shape,dtype:r}}if(t.shouldExecuteOnCPU([o])){const a=t.texData.get(o.dataId).values,[i,c,l]=E0(a,o.shape,o.dtype,r);return t.makeTensorInfo(i,c,l)}if(r==="int32")return kC(o,t);if(r==="bool"){const a=t.makeTensorInfo([],"bool",Ue("bool",1)),c=iu({inputs:{a:o,b:a},backend:t});return t.disposeIntermediateTensorInfo(a),c}throw new Error(`Error in Cast: failed to cast ${o.dtype} to ${r}`)}const NC={kernelName:gs,backendName:"webgl",kernelFunc:ho};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const oa="return ceil(x);",AC=U({opSnippet:oa,packedOpSnippet:oa,cpuKernelImpl:k0}),FC={kernelName:bo,backendName:"webgl",kernelFunc:AC};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class DC{constructor(e){this.variableNames=["A"],this.customUniforms=[{name:"minVal",type:"float"},{name:"maxVal",type:"float"}],this.outputShape=e,this.userCode=`

      void main() {
        float value = getAAtOutCoords();
        if (isnan(value)) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, minVal, maxVal));
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class OC{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"minVal",type:"float"},{name:"maxVal",type:"float"}],this.outputShape=e,this.userCode=`
      void main() {
        vec4 value = getAAtOutCoords();

        if (any(isnan(value))) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, vec4(minVal), vec4(maxVal)));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function PC(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{clipValueMin:r,clipValueMax:a}=s;let i;$().getBool("WEBGL_PACK_CLIP")?i=new OC(o.shape):i=new DC(o.shape);const c=[[r],[a]];return t.runWebGLProgram(i,[o],o.dtype,c)}const _C={kernelName:pd,backendName:"webgl",kernelFunc:PC};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class LC{constructor(e){this.variableNames=["real","imag"],this.outputShape=e,this.userCode=`
      void main() {
        float re = abs(getRealAtOutCoords());
        float im = abs(getImagAtOutCoords());
        float mx = max(re, im);

        // sadly the length function in glsl is not underflow-safe
        // (at least not on Intel GPUs). So the safe solution is
        // to ensure underflow-safety in all cases.
        setOutput(
          mx == 0.0 ? 0.0 : mx * length(vec2(1, min(re, im)/mx))
        );
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ra(n,e){return{dataId:e.dataId,dtype:e.dtype,shape:n.shape}}function BC(n){const{inputs:e,backend:t}=n,{x:s}=e,o=t.texData.get(s.dataId),r=new LC(s.shape),a=[ra(s,o.complexTensorInfos.real),ra(s,o.complexTensorInfos.imag)];return t.runWebGLProgram(r,a,a[0].dtype)}const MC={kernelName:ka,backendName:"webgl",kernelFunc:BC};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class VC{constructor(e){this.outputShape=[],this.outputShape=Tt(e,1),this.variableNames=e.map((a,i)=>`T${i}`);const t=new Array(e.length-1);t[0]=e[0][1];for(let a=1;a<t.length;a++)t[a]=t[a-1]+e[a][1];const s=[`if (yC < ${t[0]}) setOutput(getT0(yR, yC));`];for(let a=1;a<t.length;a++){const i=t[a-1];s.push(`else if (yC < ${t[a]}) setOutput(getT${a}(yR, yC-${i}));`)}const o=t.length,r=t[t.length-1];s.push(`else setOutput(getT${o}(yR, yC-${r}));`),this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int yR = coords.x;
        int yC = coords.y;

        ${s.join(`
        `)}
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class UC{constructor(e,t){this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[],this.outputShape=Tt(e,t);const s=this.outputShape,o=s.length,r=H(o),a=pe("coords",o),i=["x","y","z","w","u","v"].slice(0,o);this.variableNames=e.map((x,g)=>`T${g}`);const c=new Array(e.length-1);c[0]=e[0][t];for(let x=1;x<c.length;x++)c[x]=c[x-1]+e[x][t];const l=i[t],u=i.slice(-2),d=i.join();let h=`if (${l} < ${c[0]}) {
        return getChannel(
            getT0(${d}), vec2(${u.join()}));
        }`;for(let x=1;x<c.length;x++){const g=c[x-1];h+=`
        if (${l} < ${c[x]}  && ${l} >= ${c[x-1]}) {
          return getChannel(
            getT${x}(${Yn(i,l,g)}),
            vec2(${Yn(u,l,g)}));
        }`}const f=c.length,p=c[c.length-1];h+=`
        return getChannel(
          getT${f}(${Yn(i,l,p)}),
          vec2(${Yn(u,l,p)}));`,this.userCode=`
      float getValue(${i.map(x=>"int "+x)}) {
        ${h}
      }

      void main() {
        ${r} coords = getOutputCoords();
        vec4 result = vec4(getValue(${a}), 0., 0., 0.);

        ${a[o-1]} = ${a[o-1]} + 1;
        if (${a[o-1]} < ${s[o-1]}) {
          result.g = getValue(${a});
        }

        ${a[o-2]} = ${a[o-2]} + 1;
        if (${a[o-2]} < ${s[o-2]}) {
          result.a = getValue(${a});
        }

        ${a[o-1]} = ${a[o-1]} - 1;
        if (${a[o-2]} < ${s[o-2]} &&
            ${a[o-1]} < ${s[o-1]}) {
          result.b = getValue(${a});
        }
        setOutput(result);
      }
    `}}function Yn(n,e,t){const s=n.indexOf(e);return n.map((r,a)=>a===s?`${r} - ${t}`:r).join()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function As(n){const{inputs:e,backend:t}=n,{input:s}=e,o=t.texData.get(s.dataId);return ye({inputs:{x:o.complexTensorInfos.imag},backend:t})}const WC={kernelName:zd,backendName:"webgl",kernelFunc:As};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bn(n,e,t){const s=n[0].dtype;if(s==="complex64"){const f=n.map(b=>Gn({inputs:{input:b},backend:t})),p=n.map(b=>As({inputs:{input:b},backend:t})),x=bn(f,e,t),g=bn(p,e,t),m=ht({inputs:{real:x,imag:g},backend:t});return f.forEach(b=>t.disposeIntermediateTensorInfo(b)),p.forEach(b=>t.disposeIntermediateTensorInfo(b)),t.disposeIntermediateTensorInfo(x),t.disposeIntermediateTensorInfo(g),m}let o=t.shouldExecuteOnCPU(n);if(s==="string"&&(o=!0),o){const f=n.map(y=>{const N=[-1,A(y.shape.slice(e))];return F({inputs:{x:y},backend:t,attrs:{shape:N}})}),p=f.map(y=>({vals:t.readSync(y.dataId),shape:y.shape})),x=Tt(f.map(y=>y.shape),1),g=f[0].shape[0]===1,m=N0(p,x,s,g),b=Tt(n.map(y=>y.shape),e),C=t.makeTensorInfo(b,s,m);return f.forEach(y=>t.disposeIntermediateTensorInfo(y)),C}const r=n.filter(f=>A(f.shape)>0),a=$().getBool("WEBGL_PACK_ARRAY_OPERATIONS")&&r[0].shape.length>1;if(r.length===1){const f=a?new Be(n[0].shape,Je):new nt(n[0].shape,Je);return t.runWebGLProgram(f,n,s)}const i=$().getNumber("WEBGL_MAX_TEXTURES_IN_SHADER");if(r.length>i){const f=[];for(let x=0;x<r.length;x+=i){const g=r.slice(x,x+i);f.push(bn(g,e,t))}const p=bn(f,e,t);for(const x of f)t.disposeIntermediateTensorInfo(x);return p}if(a){const f=new UC(r.map(p=>p.shape),e);return t.runWebGLProgram(f,r,s)}const{tensors2D:c,outShape:l}=GC(r,e,t),u=new VC(c.map(f=>f.shape)),d=t.runWebGLProgram(u,c,s);c.forEach(f=>t.disposeIntermediateTensorInfo(f));const h=F({inputs:{x:d},attrs:{shape:l},backend:t});return t.disposeIntermediateTensorInfo(d),h}function GC(n,e,t){const s=Tt(n.map(r=>r.shape),e);return{tensors2D:n.map(r=>F({inputs:{x:r},attrs:{shape:[-1,A(r.shape.slice(e))]},backend:t})),outShape:s}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cu(n){const{inputs:e,backend:t,attrs:s}=n,{axis:o}=s,r=ce(o,e[0].shape)[0],a=e.map(l=>l.shape);Ui(a,r);const i=Tt(e.map(l=>l.shape),r);if(A(i)===0)return t.makeTensorInfo(i,e[0].dtype,[]);const c=e.filter(l=>A(l.shape)>0);return c.length===1?ye({inputs:{x:c[0]},backend:t}):bn(c,r,t)}const zC={kernelName:md,backendName:"webgl",kernelFunc:cu};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class lu{constructor(e,t=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.outputShape=e.outShape;const a=e.padInfo.top,i=e.padInfo.left,c=e.strideHeight,l=e.strideWidth,u=e.dilationHeight,d=e.dilationWidth,h=e.filterHeight,f=e.filterWidth,p=Math.floor(e.inChannels/4)*4,x=e.inChannels%4,g=e.dataFormat==="channelsLast",m=g?1:2,b=g?2:3,C=g?3:1;let y="",S="";s&&(o?y=`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:r?y=`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:y=`
          float activation(float x) {
            ${s}
          }
        `,S="result = activation(result);");const N=t?"result += getBiasAtOutCoords();":"";t&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${y}

      const ivec2 strides = ivec2(${c}, ${l});
      const ivec2 pads = ivec2(${a}, ${i});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d2 = coords[${C}];

        ivec2 xRCCorner =
            ivec2(coords[${m}], coords[${b}]) * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${h}; wR++) {
          int xR = xRCorner + wR * ${u};

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${f}; wC++) {
            int xC = xCCorner + wC * ${d};

            if (xC < 0 || xC >= ${e.inWidth}) {
              continue;
            }

            for (int d1 = 0; d1 < ${p}; d1 += 4) {
              vec4 wValues = vec4(
                getW(wR, wC, d1, d2),
                getW(wR, wC, d1 + 1, d2),
                getW(wR, wC, d1 + 2, d2),
                getW(wR, wC, d1 + 3, d2)
              );

              if (${g}) {
                vec4 xValues = vec4(
                  getX(batch, xR, xC, d1),
                  getX(batch, xR, xC, d1 + 1),
                  getX(batch, xR, xC, d1 + 2),
                  getX(batch, xR, xC, d1 + 3)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec4 xValues = vec4(
                  getX(batch, d1, xR, xC),
                  getX(batch, d1 + 1, xR, xC),
                  getX(batch, d1 + 2, xR, xC),
                  getX(batch, d1 + 3, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }
            }

            if (${x===1}) {

              if (${g}) {
                dotProd +=
                    getX(batch, xR, xC, ${p}) *
                    getW(wR, wC, ${p}, d2);
              } else {
                dotProd +=
                    getX(batch, ${p}, xR, xC) *
                    getW(wR, wC, ${p}, d2);
              }

            } else if (${x===2}) {
              vec2 wValues = vec2(
                getW(wR, wC, ${p}, d2),
                getW(wR, wC, ${p} + 1, d2)
              );

              if (${g}) {
                vec2 xValues = vec2(
                  getX(batch, xR, xC, ${p}),
                  getX(batch, xR, xC, ${p} + 1)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec2 xValues = vec2(
                  getX(batch, ${p}, xR, xC),
                  getX(batch, ${p} + 1, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            } else if (${x===3}) {
              vec3 wValues = vec3(
                getW(wR, wC, ${p}, d2),
                getW(wR, wC, ${p} + 1, d2),
                getW(wR, wC, ${p} + 2, d2)
              );

              if (${g}) {
                vec3 xValues = vec3(
                  getX(batch, xR, xC, ${p}),
                  getX(batch, xR, xC, ${p} + 1),
                  getX(batch, xR, xC, ${p} + 2)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec3 xValues = vec3(
                  getX(batch, ${p}, xR, xC),
                  getX(batch, ${p} + 1, xR, xC),
                  getX(batch, ${p} + 2, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            }
          }
        }

        float result = dotProd;
        ${N}
        ${S}
        setOutput(result);
      }
    `}}class HC{constructor(e){this.variableNames=["x","W"],this.outputShape=e.outShape;const t=e.padInfo.front,s=e.padInfo.top,o=e.padInfo.left,r=e.strideDepth,a=e.strideHeight,i=e.strideWidth,c=e.dilationDepth,l=e.dilationHeight,u=e.dilationWidth,d=e.filterDepth,h=e.filterHeight,f=e.filterWidth,p=Math.floor(e.inChannels/4)*4,x=e.inChannels%4;this.userCode=`
      const ivec3 strides = ivec3(${r}, ${a}, ${i});
      const ivec3 pads = ivec3(${t}, ${s}, ${o});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d2 = coords.u;

        ivec3 xFRCCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xFCorner = xFRCCorner.x;
        int xRCorner = xFRCCorner.y;
        int xCCorner = xFRCCorner.z;

        // Convolve x(?, ?, ?, d1) with w(:, :, :, d1, d2) to get
        // y(yF, yR, yC, d2). ? = to be determined. : = across all
        // values in that axis.
        float dotProd = 0.0;
        for (int wF = 0; wF < ${d}; wF++) {
          int xF = xFCorner + wF * ${c};

          if (xF < 0 || xF >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${h}; wR++) {
            int xR = xRCorner + wR * ${l};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${f}; wC++) {
              int xC = xCCorner + wC * ${u};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              for (int d1 = 0; d1 < ${p}; d1 += 4) {
                vec4 xValues = vec4(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                vec4 wValues = vec4(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (${x===1}) {
                dotProd +=
                  getX(batch, xF, xR, xC, ${p}) *
                  getW(wF, wR, wC, ${p}, d2);
              } else if (${x===2}) {
                vec2 xValues = vec2(
                  getX(batch, xF, xR, xC, ${p}),
                  getX(batch, xF, xR, xC, ${p} + 1)
                );
                vec2 wValues = vec2(
                  getW(wF, wR, wC, ${p}, d2),
                  getW(wF, wR, wC, ${p} + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (${x===3}) {
                vec3 xValues = vec3(
                  getX(batch, xF, xR, xC, ${p}),
                  getX(batch, xF, xR, xC, ${p} + 1),
                  getX(batch, xF, xR, xC, ${p} + 2)
                );
                vec3 wValues = vec3(
                  getW(wF, wR, wC, ${p}, d2),
                  getW(wF, wR, wC, ${p} + 1, d2),
                  getW(wF, wR, wC, ${p} + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class uu{constructor(e,t=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=e.outShape,this.enableShapeUniforms=de(this.outputShape.length);const a=e.padInfo.left,i=e.strideWidth,c=e.dilationWidth,l=e.filterHeight,u=e.filterWidth,d=u;let h=`
       int xR; int xC; int xCOffset;
       vec4 wTexel; vec4 previous; vec4 final;`;for(let g=0;g<u;g++)h+=`
           vec4 xTexelC${g*2};
           int xTexelC${g*2}Ready;
           vec4 xTexelC${g*2+1};
           int xTexelC${g*2+1}Ready;
           vec4 xC${g};`;h+=`
     for (int r = 0; r < ${l}; r++) {
      for (int d1 = 0; d1 < ${e.inChannels}; d1 += 2) {
       `;for(let g=0;g<u;g++)h+=`
           xTexelC${g*2} = vec4(0.0);
           xTexelC${g*2}Ready = 0;
           xTexelC${g*2+1} = vec4(0.0);
           xTexelC${g*2+1}Ready = 0;
           xC${g} = vec4(0.0);`;h+=`
         xR = xRCorner + r * dilations[0];
         if (xR >=0 && xR < inDims[0]) {
       `;for(let g=0;g<(d+1)/2;g++){const m=g*2;if(h+=`
           xC = xCCorner + ${m*c};
           `,i===1){if(m<u&&(a%2===1?(h+=`
                 xCOffset = xC + 1;
                 if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m}Ready == 0) {
                   xTexelC${m} = getX(batch, xR, xCOffset, d1);

                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${m}.zw = vec2(0.0);
                   }
                   xTexelC${m}Ready = 1;
                 }
               `,c===1&&m>0?h+=`
                 xC${m} = vec4(xTexelC${m-2}.zw, xTexelC${m}.xy);
                 `:h+=`
                   xCOffset = xC + 1 - 2;

                   if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       previous.zw = vec2(0.0);
                     }

                     xC${m} = vec4(previous.zw, xTexelC${m}.xy);
                   } else {
                     xC${m} = vec4(0.0, 0.0, xTexelC${m}.xy);
                   }
                   `):h+=`
                 if (xC >= 0 && xC < inDims[1] && xTexelC${m}Ready == 0) {
                   xTexelC${m} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${m}.zw = vec2(0.0);
                   }
                   xTexelC${m}Ready = 1;
                 }

                 xC${m} = xTexelC${m};
                 `,m+1<u)){const b=a%2===0?ps(c):c;c%2===0&&a%2===1||c%2!==0&&a%2!==1?(h+=`
                   xCOffset = xC + imod(pads[1], 2) + ${b};

                   if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m+1}Ready == 0) {
                     xTexelC${m+1} = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       xTexelC${m+1}.zw = vec2(0.0);
                     }
                     xTexelC${m+1}Ready = 1;
                   }
                   `,c>1?h+=`
                     xCOffset -= 2;
                     if (xCOffset >= 0 && xCOffset < inDims[1]) {
                      previous = getX(batch, xR, xCOffset, d1);
                      xC${m+1} = vec4(previous.zw, xTexelC${m+1}.xy);
                     } else {
                      xC${m+1} = vec4(0.0, 0.0, xTexelC${m+1}.xy);
                     }
                     `:h+=`
                     xC${m+1} = vec4(xTexelC${m}.zw, xTexelC${m+1}.xy);
                     `):b===1?h+=`
                     xC${m+1} = xTexelC${m};
                     `:h+=`
                     xCOffset = xC + ${b};

                     if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m+1}Ready == 0) {
                       xTexelC${m+1} = getX(batch, xR, xCOffset, d1);
                       if (xCOffset + 1 >= inDims[1]) {
                         xTexelC${m+1}.zw = vec2(0.0);
                       }
                       xTexelC${m+1}Ready = 1;
                     }

                     xC${m+1} = xTexelC${m+1};
                     `}}else m<u&&(a%2===1?(h+=`
                 xCOffset = xC + 1 - strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m}Ready == 0) {
                   xTexelC${m} = getX(batch, xR, xCOffset, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${m}.zw = vec2(0.0);
                   }
                   xTexelC${m}Ready = 1;
                 }

                 if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${m+1}Ready == 0) {
                   xTexelC${m+1} = getX(batch, xR, xC + 1, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xC + 2 >= inDims[1]) {
                     xTexelC${m+1}.zw = vec2(0.0);
                   }
                   xTexelC${m+1}Ready = 1;
                 }

                 xC${m} = vec4(xTexelC${m}.zw, xTexelC${m+1}.zw);
               `,m+1<u&&(h+=`
                   final = vec4(0.0);
                   xCOffset = xC + 1 + strides[1];
                   if(xCOffset >= 0 && xCOffset < inDims[1]) {
                     final = getX(batch, xR, xCOffset, d1);
                   }
                   xC${m+1} = vec4(xTexelC${m+1}.xy, final.xy);
                 `)):(h+=`
                 if(xC >= 0 && xC < inDims[1] && xTexelC${m}Ready == 0) {
                   xTexelC${m} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${m}.zw = vec2(0.0);
                   }
                   xTexelC${m}Ready = 1;
                 }

                 xCOffset = xC + strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m+1}Ready == 0) {
                   xTexelC${m+1} = getX(batch, xR, xCOffset, d1);
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${m+1}.zw = vec2(0.);
                   }
                   xTexelC${m+1}Ready = 1;
                 }

                 xC${m} = vec4(
                   xTexelC${m}.xy, xTexelC${m+1}.xy);
               `,m+1<u&&(h+=`
                   xC${m+1} = vec4(xTexelC${m}.zw, xTexelC${m+1}.zw);
                 `)));m<u&&(h+=`
             wTexel = getW(r, ${m}, d1, d2);
             dotProd += xC${m}.xxzz * vec4(wTexel.xy, wTexel.xy);
             if(d1 + 1 < ${e.inChannels}) {
               dotProd += xC${m}.yyww * vec4(wTexel.zw, wTexel.zw);
             }
           `,m+1<u&&(h+=`
               wTexel = getW(r, ${m+1}, d1, d2);
               dotProd += xC${m+1}.xxzz * vec4(wTexel.xy, wTexel.xy);
               if(d1 + 1 < ${e.inChannels}) {
                 dotProd += xC${m+1}.yyww * vec4(wTexel.zw, wTexel.zw);
               }
             `))}h+=`
     }
   `,h+=`
     }
   `,h+=`
     }
   `;let f="",p="";s&&(o?f=`vec4 activation(vec4 a) {
           vec4 b = getPreluActivationWeightsAtOutCoords();
           ${s}
         }`:r?f=`vec4 activation(vec4 a) {
           vec4 b = getLeakyreluAlphaAtOutCoords();
           ${s}
         }`:f=`vec4 activation(vec4 x) {
           ${s}
         }`,p="result = activation(result);");const x=t?"result += getBiasAtOutCoords();":"";t&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
       ${f}

       void main() {
         ivec4 coords = getOutputCoords();
         int batch = coords.x;
         ivec2 xRCCorner = coords.yz * strides - pads;
         int d2 = coords.w;
         int xRCorner = xRCCorner.x;
         int xCCorner = xRCCorner.y;

         //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
         vec4 dotProd = vec4(0.000000000000001);

         ${h}

         vec4 result = dotProd - vec4(0.000000000000001);
         ${x}
         ${p}
         setOutput(result);
       }
     `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class XC{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"inputShape",type:"ivec4"},{name:"pad",type:"ivec2"},{name:"stride",type:"ivec2"},{name:"dilation",type:"ivec2"},{name:"inChannels",type:"int"},{name:"itemsPerBlockRow",type:"int"},{name:"outWidth",type:"int"}],this.outputShape=e,this.enableShapeUniforms=de(this.outputShape.length);const{dataFormat:s}=t,o=ge(),r=s==="channelsLast",a=r?1:2,i=r?2:3,c=this.enableShapeUniforms?"if(blockIndex < outShape[2] && pos < outShape[1]) {":`if(blockIndex < ${e[2]} && pos < ${e[1]}) {`;let l="";for(let u=0;u<=1;u++)for(let d=0;d<=1;d++)l+=`
          blockIndex = rc.z + ${d};
          pos = rc.y + ${u};

          ${c}
            offsetY = int(blockIndex / outWidth) * stride[0] - pad[0];
            d0 = offsetY + dilation[0] * (pos / itemsPerBlockRow);

            if(d0 < inputShape[${a}] && d0 >= 0) {
              // Use custom imod instead mod. On Intel GPU, mod may generate
              // unexpected value.
              // https://github.com/tensorflow/tfjs/issues/5447
              offsetX = imod(blockIndex, outWidth) * stride[1] - pad[1];
              d1 = offsetX + dilation[1] * (imod(pos, itemsPerBlockRow) /
                  inChannels);

              if(d1 < inputShape[${i}] && d1 >= 0) {

                ch = imod(pos, inChannels);

                if (${r}) {
                  innerDims = vec2(d1, ch);
                  result[${u*2+d}] = getChannel(
                    getA(rc.x, d0, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                } else {
                  innerDims = vec2(d0, d1);
                  result[${u*2+d}] = getChannel(
                    getA(rc.x, ch, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                }
              }
            }
          }
        `;this.userCode=`
      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0);

        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;
        vec2 innerDims;

        ${l}

        ${o.output} = result;
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hs(n,e){const t=n.length;return t>=3?e?[...n.slice(0,-3),n[t-3]*n[t-2],n[t-1]]:[...n.slice(0,-3),n[t-3],n[t-2]*n[t-1]]:!e&&t===1&&n[0]>1?[n[0],1]:null}function du({x:n,filter:e,convInfo:t,backend:s,bias:o=null,preluActivationWeights:r=null,leakyreluAlpha:a=0,activation:i=null}){const c=n.shape,l=s.texData.get(n.dataId),u=t.inChannels,d=c[0]*c[1]*c[2],h=t.outChannels,f=t.dataFormat==="channelsLast",p=!1,x=!1;let g;const m=[];if(r!=null){const y=hs(r.shape,f);y!=null&&(r=F({inputs:{x:r},backend:s,attrs:{shape:y}}),m.push(r))}if(o!=null){const y=hs(o.shape,f);y!=null&&(o=F({inputs:{x:o},backend:s,attrs:{shape:y}}),m.push(o))}if(!((d===1||h===1)&&u>su)&&l.isPacked&&f&&l.texture!=null&&c[2]%2!==0&&oe(l.shape.slice(-3),c.slice(-3))){const y=c[0]*c[1]*(c[2]+1),S={dataId:n.dataId,shape:[1,y,t.inChannels],dtype:n.dtype},N=l.shape;l.shape=l.shape.slice(),l.shape[l.shape.length-2]++,R(Rn(l.shape,S.shape),()=>`packed reshape ${l.shape} to ${S.shape} isn't free`);const k=F({inputs:{x:e},backend:s,attrs:{shape:[1,t.inChannels,t.outChannels]}});m.push(k);const T=ds({a:S,b:k,backend:s,transposeA:p,transposeB:x,bias:o,activation:i,preluActivationWeights:r,leakyreluAlpha:a}),I=s.texData.get(T.dataId);R(I.isPacked,()=>"batchMatMul result is expected to be packed"),l.shape=N,I.shape=t.outShape,g=ye({inputs:{x:T},backend:s}),g.shape=t.outShape,m.push(T)}else{const y=t.outHeight*t.outWidth,S=F({inputs:{x:n},backend:s,attrs:{shape:f?[t.batchSize,y,t.inChannels]:[t.batchSize,t.inChannels,y]}}),N=F({inputs:{x:e},backend:s,attrs:{shape:[1,t.inChannels,t.outChannels]}}),k=ds({a:f?S:N,b:f?N:S,transposeA:!f,transposeB:x,backend:s,bias:o,activation:i,preluActivationWeights:r,leakyreluAlpha:a});g=F({inputs:{x:k},backend:s,attrs:{shape:t.outShape}}),m.push(S),m.push(N),m.push(k)}for(const y of m)s.disposeIntermediateTensorInfo(y);return g}function hu({x:n,filter:e,convInfo:t,backend:s,bias:o=null,preluActivationWeights:r=null,leakyreluAlpha:a=0,activation:i=null}){const{filterWidth:c,filterHeight:l,inChannels:u,outWidth:d,outHeight:h,dataFormat:f}=t,p=f==="channelsLast",x=c*l*u,g=h*d,m=[t.batchSize,x,g],b=!0,C=!1,y=[];if(r!=null){const G=hs(r.shape,p);G!=null&&(r=F({inputs:{x:r},backend:s,attrs:{shape:G}}),y.push(r))}if(o!=null){const G=hs(o.shape,p);G!=null&&(o=F({inputs:{x:o},backend:s,attrs:{shape:G}}),y.push(o))}const S=F({inputs:{x:e},backend:s,attrs:{shape:[1,x,A(e.shape)/x]}});y.push(S);const N=new XC(m,t),k=[n.shape,[t.padInfo.top,t.padInfo.left],[t.strideHeight,t.strideWidth],[t.dilationHeight,t.dilationWidth],[t.inChannels],[t.filterWidth*t.inChannels],[t.outWidth]],T=s.runWebGLProgram(N,[n],"float32",k),I=F({inputs:{x:T},backend:s,attrs:{shape:m}});y.push(T),y.push(I);const v=o!=null,w=r!=null,E=i==="leakyrelu",_=i?kn(i,!0):null,B=new nu(p?I.shape:S.shape,p?S.shape:I.shape,p?[t.batchSize,g,t.outChannels]:[t.batchSize,t.outChannels,g],b,C,v,_,w,E),L=p?[I,S]:[S,I];if(o&&L.push(o),w&&L.push(r),E){const G=s.makeTensorInfo([],"float32",Nt(a,"float32"));L.push(G),y.push(G)}const W=s.runWebGLProgram(B,L,"float32"),j=F({inputs:{x:W},backend:s,attrs:{shape:t.outShape}});y.push(W);for(const G of y)s.disposeIntermediateTensorInfo(G);return j}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qC(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,filter:r}=e,{strides:a,pad:i,dataFormat:c,dilations:l,dimRoundingMode:u}=s,d=en(c),h=_e(o.shape,r.shape,a,l,i,u,!1,d);let f;if(h.filterHeight===1&&h.filterWidth===1&&h.dilationHeight===1&&h.dilationWidth===1&&h.strideHeight===1&&h.strideWidth===1&&(h.padInfo.type==="SAME"||h.padInfo.type==="VALID"))f=du({x:o,filter:r,convInfo:h,backend:t});else if(h.strideWidth<=2&&d==="channelsLast"&&$().getBool("WEBGL_EXP_CONV")){const x=new uu(h),g=[[h.padInfo.top,h.padInfo.left],[h.strideHeight,h.strideWidth],[h.dilationHeight,h.dilationWidth],[h.inHeight,h.inWidth]];f=t.runWebGLProgram(x,[o,r],"float32",g)}else if($().getBool("WEBGL_CONV_IM2COL"))f=hu({x:o,filter:r,convInfo:h,backend:t});else{const x=new lu(h);f=t.runWebGLProgram(x,[o,r],"float32")}const p=F({inputs:{x:f},backend:t,attrs:{shape:h.outShape}});return t.disposeIntermediateTensorInfo(f),p}const jC={kernelName:gd,backendName:"webgl",kernelFunc:qC};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class KC{constructor(e){this.variableNames=["x","dy"],this.outputShape=e.filterShape;const t=e.strideHeight,s=e.strideWidth,o=e.padInfo.top,r=e.padInfo.left,a=e.dataFormat==="channelsLast";this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int d2 = coords.w;

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${o};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${s} - ${r};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              ${a?`float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);`:`float dyValue = getDy(b, d2, yR, yC);
              float xValue = getX(b, d1, xR, xC);
              dotProd += (xValue * dyValue);`}
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class YC{constructor(e){this.variableNames=["dy","W"],this.outputShape=e.inShape;const t=e.filterHeight,s=e.filterWidth,o=e.strideHeight,r=e.strideWidth,a=e.dataFormat==="channelsLast",i=t-1-e.padInfo.top,c=s-1-e.padInfo.left,l=a?1:2,u=a?2:3,d=a?3:1;this.userCode=`
      const ivec2 pads = ivec2(${i}, ${c});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[${d}];

        ivec2 dyCorner = ivec2(coords[${l}], coords[${u}]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${o}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            float dyC = float(dyCCorner + wC) / ${r}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${s} - 1 - wC;

            for (int d2 = 0; d2 < ${e.outChannels}; d2++) {

              if (${a}) {
                float xValue = getDy(batch, idyR, idyC, d2);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              } else {
                float xValue = getDy(batch, d2, idyR, idyC);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }

            }
          }
        }
        setOutput(dotProd);
      }
    `}}class QC{constructor(e){this.variableNames=["x","dy"],this.outputShape=e.filterShape;const t=e.strideDepth,s=e.strideHeight,o=e.strideWidth,r=e.padInfo.front,a=e.padInfo.top,i=e.padInfo.left;this.userCode=`
      void main() {
        ivec5 coords = getOutputCoords();
        int wF = coords.x;
        int wR = coords.y;
        int wC = coords.z;
        int d1 = coords.w;
        int d2 = coords.u;

        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yF = 0; yF < ${e.outDepth}; yF++) {
            int xF = wF + yF * ${t} - ${r};

            if (xF < 0 || xF >= ${e.inDepth}) {
              continue;
            }

            for (int yR = 0; yR < ${e.outHeight}; yR++) {
              int xR = wR + yR * ${s} - ${a};

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int yC = 0; yC < ${e.outWidth}; yC++) {
                int xC = wC + yC * ${o} - ${i};

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float dyValue = getDy(b, yF, yR, yC, d2);
                float xValue = getX(b, xF, xR, xC, d1);
                dotProd += (xValue * dyValue);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class ZC{constructor(e){this.variableNames=["dy","W"],this.outputShape=e.inShape;const t=e.filterDepth,s=e.filterHeight,o=e.filterWidth,r=e.strideDepth,a=e.strideHeight,i=e.strideWidth,c=t-1-e.padInfo.front,l=s-1-e.padInfo.top,u=o-1-e.padInfo.left;this.userCode=`
      const ivec3 pads = ivec3(${c}, ${l}, ${u});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.u;


        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyFCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        float dotProd = 0.0;
        for (int wF = 0; wF < ${t}; wF++) {
          float dyF = float(dyFCorner + wF) / ${r}.0;

          if (dyF < 0.0 || dyF >= ${e.outDepth}.0 || fract(dyF) > 0.0) {
            continue;
          }
          int idyF = int(dyF);

          int wFPerm = ${t} - 1 - wF;

          for (int wR = 0; wR < ${s}; wR++) {
            float dyR = float(dyRCorner + wR) / ${a}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
              fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            int wRPerm = ${s} - 1 - wR;

            for (int wC = 0; wC < ${o}; wC++) {
              float dyC = float(dyCCorner + wC) / ${i}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              int wCPerm = ${o} - 1 - wC;

              for (int d2 = 0; d2 < ${e.outChannels}; d2++) {
                float xValue = getDy(batch, idyF, idyR, idyC, d2);
                float wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function JC(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,dy:r}=e,{strides:a,pad:i,dataFormat:c,dimRoundingMode:l,filterShape:u}=s,d=en(c),h=_e(o.shape,u,a,1,i,l,!1,d),f=new KC(h);return t.runWebGLProgram(f,[o,r],"float32")}const ey={kernelName:xd,backendName:"webgl",kernelFunc:JC};/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ty{constructor(e){this.variableNames=["dy","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"strides",type:"vec2"}],this.outputShape=e.inShape,this.enableShapeUniforms=de(this.outputShape.length);const t=e.filterHeight,s=e.filterWidth,o=t-1-e.padInfo.top,r=s-1-e.padInfo.left;this.userCode=`
      const ivec2 pads = ivec2(${o}, ${r});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];

        ivec2 dyCorner = ivec2(coords[1], coords[2]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        vec4 result = vec4(0.);
        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / strides[0];
          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);
          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            int wCPerm = ${s} - 1 - wC;

            float dyC = float(dyCCorner + wC) / strides[1];
            bool idyCVal = (dyC >= 0.0) && (dyC < ${e.outWidth}.0)
              && (fract(dyC) == 0.0);
            int idyC = int(dyC);

            float dyC2 = float(dyCCorner + wC + 1) / strides[1];
            bool idyCVal2 = (dyC2 >= 0.0) && (dyC2 < ${e.outWidth}.0)
              && (fract(dyC2) == 0.0);
            int idyC2 = int(dyC2);

            if (idyCVal && idyCVal2) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec4 dySample2 = (idyC / 2 == idyC2 / 2) ?
                  dySample : getDy(batch, idyR, idyC2, d2);

                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));

                dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample2.xy : dySample2.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal2) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC2, d2);
                vec2 dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            }
          }
        }
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ny(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,filter:r}=e,{inputShape:a,strides:i,pad:c,dataFormat:l,dimRoundingMode:u}=s,d=en(l),h=_e(a,r.shape,i,1,c,u,!1,d);if($().getBool("WEBGL_PACK_CONV2DTRANSPOSE")&&d==="channelsLast"){const f=[[h.strideHeight,h.strideWidth]],p=new ty(h);return t.runWebGLProgram(p,[o,r],"float32",f)}else{const f=new YC(h);return t.runWebGLProgram(f,[o,r],"float32")}}const sy={kernelName:bd,backendName:"webgl",kernelFunc:ny};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oy(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,filter:r}=e,{strides:a,pad:i,dilations:c}=s,l=Bn(o.shape,r.shape,a,c,i),u=new HC(l);return t.runWebGLProgram(u,[o,r],"float32")}const ry={kernelName:wd,backendName:"webgl",kernelFunc:oy};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ay(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,dy:r}=e,{strides:a,pad:i,filterShape:c}=s,l=Bn(o.shape,c,a,1,i),u=new QC(l);return t.runWebGLProgram(u,[o,r],"float32")}const iy={kernelName:Cd,backendName:"webgl",kernelFunc:ay};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cy(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,filter:r}=e,{pad:a,strides:i,inputShape:c}=s,l=Bn(c,r.shape,i,1,a),u=new ZC(l);return t.runWebGLProgram(u,[o,r],"float32")}const ly={kernelName:yd,backendName:"webgl",kernelFunc:cy};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uy=ln+`
  return cos(x);
`,dy=`
  vec4 result = cos(x);
  bvec4 isNaN = isnan(x);
  ${Pt}
  return result;
`,hy=U({opSnippet:uy,packedOpSnippet:dy}),fy={kernelName:$d,backendName:"webgl",kernelFunc:hy};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const py=`
  float e2x = exp(-x);
  return (e2x + 1.0 / e2x) / 2.0;
`,my=U({opSnippet:py}),gy={kernelName:vd,backendName:"webgl",kernelFunc:my};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class xy{constructor(e,t,s,o,r){this.variableNames=["Image","Boxes","BoxInd"],this.outputShape=[];const[a,i,c,l]=e,[u]=t,[d,h]=s;this.outputShape=[u,d,h,l];const f=o==="bilinear"?1:0,[p,x]=[`${i-1}.0`,`${c-1}.0`],[g,m,b]=d>1?[`${(i-1)/(d-1)}`,"(y2-y1) * height_ratio",`y1*${p} + float(y)*(height_scale)`]:["0.0","0.0",`0.5 * (y1+y2) * ${p}`],[C,y,S]=h>1?[`${(c-1)/(h-1)}`,"(x2-x1) * width_ratio",`x1*${x} + float(x)*(width_scale)`]:["0.0","0.0",`0.5 * (x1+x2) * ${x}`];this.userCode=`
      const float height_ratio = float(${g});
      const float width_ratio = float(${C});
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int y = coords[1];
        int x = coords[2];
        int d = coords[3];

        // get box vals
        float y1 = getBoxes(b,0);
        float x1 = getBoxes(b,1);
        float y2 = getBoxes(b,2);
        float x2 = getBoxes(b,3);

        // get image in batch index
        int bInd = round(getBoxInd(b));
        if(bInd < 0 || bInd >= ${a}) {
          return;
        }

        float height_scale = ${m};
        float width_scale = ${y};

        float in_y = ${b};
        if( in_y < 0.0 || in_y > ${p} ) {
          setOutput(float(${r}));
          return;
        }
        float in_x = ${S};
        if( in_x < 0.0 || in_x > ${x} ) {
          setOutput(float(${r}));
          return;
        }

        vec2 sourceFracIndexCR = vec2(in_x,in_y);
        if(${f} == 1) {
          // Compute the four integer indices.
          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);
          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));

          float topLeft = getImage(b, sourceFloorCR.y, sourceFloorCR.x, d);
          float bottomLeft = getImage(b, sourceCeilCR.y, sourceFloorCR.x, d);
          float topRight = getImage(b, sourceFloorCR.y, sourceCeilCR.x, d);
          float bottomRight = getImage(b, sourceCeilCR.y, sourceCeilCR.x, d);

          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);

          float top = topLeft + (topRight - topLeft) * fracCR.x;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          float newValue = top + (bottom - top) * fracCR.y;
          setOutput(newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          ivec2 sourceNearestCR = ivec2(floor(
            sourceFracIndexCR + vec2(0.5,0.5)));
          float newValue = getImage(b, sourceNearestCR.y, sourceNearestCR.x, d);
          setOutput(newValue);
        }
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const by=n=>{const{inputs:e,backend:t,attrs:s}=n,{image:o,boxes:r,boxInd:a}=e,{cropSize:i,method:c,extrapolationValue:l}=s,u=new xy(o.shape,r.shape,i,c,l);return t.runWebGLProgram(u,[o,r,a],"float32")},wy={kernelName:Td,backendName:"webgl",kernelFunc:by};var An;(function(n){n.Prod="*",n.Sum="+"})(An||(An={}));class aa{constructor(e,t,s,o){this.op=e,this.outputShape=t,this.variableNames=["x"],this.customUniforms=[{name:"index",type:"float"}];const r=this.outputShape.length,a=this.op===An.Prod?"1.0":"0.0",i=s?a:`getX(${ia(r,"coords",this.op)})`,c=this.outputShape[this.outputShape.length-1];let l="",u="";s?(l=o?`end != ${c-1}`:"end != 0",u=o?"end + 1":"end - 1"):(l=o?`end + pow2 < ${c}`:"end >= pow2",u=o?"end + pow2":"end - pow2"),this.userCode=`
      void main() {
        ${H(r)} coords = getOutputCoords();
        int end = ${ca(r,"coords",this.op)};
        float val = ${i};
        int pow2 = int(pow(2.0, index));
        if (${l}) {
          int idx = ${u};
          ${ca(r,"coords",this.op)} = idx;
          val ${this.op}= getX(${ia(r,"coords",this.op)});
        }
        setOutput(val);
      }
    `}}function ia(n,e,t){if(n===1)return`${e}`;if(n===2)return`${e}.x, ${e}.y`;if(n===3)return`${e}.x, ${e}.y, ${e}.z`;if(n===4)return`${e}.x, ${e}.y, ${e}.z, ${e}.w`;throw new Error(`Cumulative ${t} for rank ${n} is not yet supported`)}function ca(n,e,t){if(n===1)return`${e}`;if(n===2)return`${e}.y`;if(n===3)return`${e}.z`;if(n===4)return`${e}.w`;throw new Error(`Cumulative ${t} for rank ${n} is not yet supported`)}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fu(n,e,t,s,o,r){const a=e.shape.length,i=Re([s],a);let c=e;i!=null&&(c=me({inputs:{x:e},backend:t,attrs:{perm:i}}));const l=Ee(1,a)[0];if(l!==a-1)throw new Error(`WebGL cumprod shader expects an inner-most axis=${e.shape.length-1} but got axis=${s}`);const u=c.shape[l];let d=ye({inputs:{x:c},backend:t});for(let h=0;h<=Math.ceil(Math.log2(u))-1;h++){const f=new aa(n,c.shape,!1,r),p=[[h]],x=d;d=t.runWebGLProgram(f,[d],d.dtype,p),t.disposeIntermediateTensorInfo(x)}if(o){const h=new aa(n,c.shape,o,r),f=d;d=t.runWebGLProgram(h,[d],d.dtype),t.disposeIntermediateTensorInfo(f)}if(i!=null){const h=zo(i),f=me({inputs:{x:d},backend:t,attrs:{perm:h}});return t.disposeIntermediateTensorInfo(d),t.disposeIntermediateTensorInfo(c),f}return d}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cy(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,exclusive:a,reverse:i}=s;return fu(An.Prod,o,t,r,a,i)}const yy={kernelName:Sd,backendName:"webgl",kernelFunc:Cy};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $y(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,exclusive:a,reverse:i}=s;return fu(An.Sum,o,t,r,a,i)}const vy={kernelName:Id,backendName:"webgl",kernelFunc:$y};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sy(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,weights:r}=e,{size:a,binaryOutput:i}=s;if(o.shape.length===1){const c=t.readSync(o.dataId),l=t.readSync(r.dataId),u=Xl(c,l,r.dtype,r.shape,a);return t.makeTensorInfo([a],r.dtype,u)}else if(o.shape.length===2){const c=t.bufferSync(o),l=t.bufferSync(r),u=T0(c,l,a,i);return t.makeTensorInfo(u.shape,r.dtype,u.values)}throw new Error(`Error in denseBincount: input must be at most rank 2, but got rank${o.shape.length}.`)}const Iy={kernelName:Rd,backendName:"webgl",kernelFunc:Sy};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ty{constructor(e,t,s){this.variableNames=["x"],this.outputShape=[],this.outputShape=e,this.blockSize=t,this.dataFormat=s,this.userCode=`
    void main() {
      ivec4 coords = getOutputCoords();
      int b = coords[0];
      int h = ${this.getHeightCoordString()};
      int w = ${this.getWidthCoordString()};
      int d = ${this.getDepthCoordString()};

      int in_h = h / ${t};
      int offset_h = imod(h, ${t});
      int in_w = w / ${t};
      int offset_w = imod(w, ${t});
      int offset_d = (offset_h * ${t} + offset_w) *
        ${this.getOutputDepthSize()};
      int in_d = d + offset_d;

      float result = ${this.getInputSamplingString()};
      setOutput(result);
    }
  `}getHeightCoordString(){return this.dataFormat==="NHWC"?"coords[1]":"coords[2]"}getWidthCoordString(){return this.dataFormat==="NHWC"?"coords[2]":"coords[3]"}getDepthCoordString(){return this.dataFormat==="NHWC"?"coords[3]":"coords[1]"}getOutputDepthSize(){return this.dataFormat==="NHWC"?this.outputShape[3]:this.outputShape[1]}getInputSamplingString(){return this.dataFormat==="NHWC"?"getX(b, in_h, in_w, in_d)":"getX(b, in_d, in_h, in_w)"}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ry(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{blockSize:r,dataFormat:a}=s,i=o.shape[0],c=a==="NHWC"?o.shape[1]:o.shape[2],l=a==="NHWC"?o.shape[2]:o.shape[3],u=a==="NHWC"?o.shape[3]:o.shape[1],d=c*r,h=l*r,f=u/(r*r),p=a==="NHWC"?[i,d,h,f]:[i,f,d,h],x=new Ty(p,r,a);return t.runWebGLProgram(x,[o],o.dtype)}const Ey={kernelName:Ed,backendName:"webgl",kernelFunc:Ry};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class pu{constructor(e,t=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=e.outShape,this.enableShapeUniforms=de(this.outputShape.length);const a=e.filterHeight,i=e.filterWidth,c=e.outChannels/e.inChannels;let l="",u="";s&&(o?l=`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:r?l=`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:l=`
          float activation(float x) {
            ${s}
          }
        `,u="result = activation(result);");const d=t?"result += getBiasAtOutCoords();":"";t&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${l}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${c};
        int q = d2 - d1 * ${c};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        // TO DO(dsmilkov): Flatten the two for loops and vec4 the operations.
        for (int wR = 0; wR < ${a}; wR++) {
          int xR = xRCorner + wR * dilations[0];

          if (xR < 0 || xR >= inDims[0]) {
            continue;
          }

          for (int wC = 0; wC < ${i}; wC++) {
            int xC = xCCorner + wC * dilations[1];

            if (xC < 0 || xC >= inDims[1]) {
              continue;
            }

            float xVal = getX(batch, xR, xC, d1);
            float wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
          }
        }

        float result = dotProd;
        ${d}
        ${u}
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class mu{constructor(e,t=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=e.outShape,this.enableShapeUniforms=de(this.outputShape.length);const a=e.outChannels/e.inChannels,i=e.padInfo.left,c=e.strideWidth,l=e.dilationWidth,u=e.filterHeight,d=e.filterWidth,h=d;let f=`
      int xR; int xC; int xCOffset;
      vec4 wTexel; vec4 previous; vec4 final;`;for(let m=0;m<d;m++)f+=`
          vec4 xTexelC${m*2};
          int xTexelC${m*2}Ready;
          vec4 xTexelC${m*2+1};
          int xTexelC${m*2+1}Ready;
          vec4 xC${m};`;f+=`
    for (int r = 0; r < ${u}; r++) {
      `;for(let m=0;m<d;m++)f+=`
          xTexelC${m*2} = vec4(0.0);
          xTexelC${m*2}Ready = 0;
          xTexelC${m*2+1} = vec4(0.0);
          xTexelC${m*2+1}Ready = 0;
          xC${m} = vec4(0.0);`;f+=`
        xR = xRCorner + r * dilations[0];
        if (xR >=0 && xR < inDims[0]) {
      `;for(let m=0;m<(h+1)/2;m++){const b=m*2;if(f+=`
          xC = xCCorner + ${b*l};
          `,c===1){if(b<d&&(i%2===1?(f+=`
                xCOffset = xC + 1;
                if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b}Ready == 0) {
                  xTexelC${b} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${b}.zw = vec2(0.0);
                  }
                  xTexelC${b}Ready = 1;
                }
              `,l===1&&b>0?f+=`
                xC${b} = vec4(xTexelC${b-2}.zw, xTexelC${b}.xy);
                `:f+=`
                  xCOffset = xC + 1 - 2;

                  if (xCOffset >= 0 && xCOffset < inDims[1]) {
                    previous = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      previous.zw = vec2(0.0);
                    }

                    xC${b} = vec4(previous.zw, xTexelC${b}.xy);
                  } else {
                    xC${b} = vec4(0.0, 0.0, xTexelC${b}.xy);
                  }
                  `):f+=`
                if (xC >= 0 && xC < inDims[1] && xTexelC${b}Ready == 0) {
                  xTexelC${b} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${b}.zw = vec2(0.0);
                  }
                  xTexelC${b}Ready = 1;
                }

                xC${b} = xTexelC${b};
                `,b+1<d)){const C=i%2===0?ps(l):l;l%2===0&&i%2===1||l%2!==0&&i%2!==1?(f+=`
                  xCOffset = xC + imod(pads[1], 2) + ${C};

                  if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b+1}Ready == 0) {
                    xTexelC${b+1} = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      xTexelC${b+1}.zw = vec2(0.0);
                    }
                    xTexelC${b+1}Ready = 1;
                  }
                  `,l>1?f+=`
                    xCOffset -= 2;
                    if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);
                     xC${b+1} = vec4(previous.zw, xTexelC${b+1}.xy);
                    } else {
                     xC${b+1} = vec4(0.0, 0.0, xTexelC${b+1}.xy);
                    }
                    `:f+=`
                    xC${b+1} = vec4(xTexelC${b}.zw, xTexelC${b+1}.xy);
                    `):C===1?f+=`
                    xC${b+1} = xTexelC${b};
                    `:f+=`
                    xCOffset = xC + ${C};

                    if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b+1}Ready == 0) {
                      xTexelC${b+1} = getX(batch, xR, xCOffset, d1);
                      if (xCOffset + 1 >= inDims[1]) {
                        xTexelC${b+1}.zw = vec2(0.0);
                      }
                      xTexelC${b+1}Ready = 1;
                    }

                    xC${b+1} = xTexelC${b+1};
                    `}}else b<d&&(i%2===1?(f+=`
                xCOffset = xC + 1 - strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b}Ready == 0) {
                  xTexelC${b} = getX(batch, xR, xCOffset, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${b}.zw = vec2(0.0);
                  }
                  xTexelC${b}Ready = 1;
                }

                if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${b+1}Ready == 0) {
                  xTexelC${b+1} = getX(batch, xR, xC + 1, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xC + 2 >= inDims[1]) {
                    xTexelC${b+1}.zw = vec2(0.0);
                  }
                  xTexelC${b+1}Ready = 1;
                }

                xC${b} = vec4(xTexelC${b}.zw, xTexelC${b+1}.zw);
              `,b+1<d&&(f+=`
                  final = vec4(0.0);
                  xCOffset = xC + 1 + strides[1];
                  if(xCOffset >= 0 && xCOffset < inDims[1]) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xC${b+1} = vec4(xTexelC${b+1}.xy, final.xy);
                `)):(f+=`
                if(xC >= 0 && xC < inDims[1] && xTexelC${b}Ready == 0) {
                  xTexelC${b} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${b}.zw = vec2(0.0);
                  }
                  xTexelC${b}Ready = 1;
                }

                xCOffset = xC + strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b+1}Ready == 0) {
                  xTexelC${b+1} = getX(batch, xR, xCOffset, d1);
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${b+1}.zw = vec2(0.);
                  }
                  xTexelC${b+1}Ready = 1;
                }

                xC${b} = vec4(
                  xTexelC${b}.xy, xTexelC${b+1}.xy);
              `,b+1<d&&(f+=`
                  xC${b+1} = vec4(xTexelC${b}.zw, xTexelC${b+1}.zw);
                `)));b<d&&(f+=`
            wTexel = getW(r, ${b}, d1, q);
            dotProd += xC${b} * vec4(wTexel.xz, wTexel.xz);
          `,b+1<d&&(f+=`
              wTexel = getW(r, ${b+1}, d1, q);
              dotProd += xC${b+1} * vec4(wTexel.xz, wTexel.xz);
            `))}f+=`
    }
  `,f+=`
      }
    `;let p="",x="";s&&(o?p=`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:r?p=`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:p=`vec4 activation(vec4 x) {
          ${s}
        }`,x="result = activation(result);");const g=t?"result += getBiasAtOutCoords();":"";t&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${p}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${a};
        int q = d2 - d1 * ${a};
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
        vec4 dotProd = vec4(0.000000000000001);

        ${f}

        vec4 result = dotProd - vec4(0.000000000000001);
        ${g}
        ${x}
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ky(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,filter:r}=e,{strides:a,pad:i,dilations:c,dimRoundingMode:l}=s;let u=c;u==null&&(u=[1,1]),R(Jt(a,u),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${a} and dilations '${u}'`);const d=_e(o.shape,r.shape,a,u,i,l,!0);let h;$().getBool("WEBGL_PACK_DEPTHWISECONV")&&d.strideWidth<=2&&d.outChannels/d.inChannels===1?h=new mu(d):h=new pu(d);const f=[[d.padInfo.top,d.padInfo.left],[d.strideHeight,d.strideWidth],[d.dilationHeight,d.dilationWidth],[d.inHeight,d.inWidth]];return t.runWebGLProgram(h,[o,r],"float32",f)}const Ny={kernelName:kd,backendName:"webgl",kernelFunc:ky};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ay{constructor(e){this.variableNames=["x","dy"],this.outputShape=e.filterShape;const t=e.strideHeight,s=e.strideWidth,o=e.padInfo.top,r=e.padInfo.left,a=e.outChannels/e.inChannels;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int dm = coords.w;
        int d2 = d1 * ${a} + dm;

        float dotProd = 0.0;

        // TO DO: Vec4 over the batch size
        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${o};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${s} - ${r};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class Fy{constructor(e){this.variableNames=["dy","W"],this.outputShape=e.inShape;const t=e.filterHeight,s=e.filterWidth,o=e.strideHeight,r=e.strideWidth,a=t-1-e.padInfo.top,i=s-1-e.padInfo.left,c=e.outChannels/e.inChannels;this.userCode=`
      const ivec2 pads = ivec2(${a}, ${i});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];
        ivec2 dyCorner = coords.yz - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        float dotProd = 0.0;

        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${o}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            float dyC = float(dyCCorner + wC) / ${r}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${s} - 1 - wC;

            // TO DO: Vec4 over the channelMul
            for (int dm = 0; dm < ${c}; dm++) {
              int d2 = d1 * ${c} + dm;
              float xValue = getDy(batch, idyR, idyC, d2);
              float wValue = getW(wRPerm, wCPerm, d1, dm);
              dotProd += xValue * wValue;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dy(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,dy:r}=e,{strides:a,dilations:i,pad:c,dimRoundingMode:l,filterShape:u}=s,d=_e(o.shape,u,a,i,c,l,!0),h=new Ay(d);return t.runWebGLProgram(h,[o,r],"float32")}const Oy={kernelName:Nd,backendName:"webgl",kernelFunc:Dy};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Py(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,filter:r}=e,{strides:a,dilations:i,pad:c,dimRoundingMode:l,inputShape:u}=s,d=_e(u,r.shape,a,i,c,l,!0),h=new Fy(d);return t.runWebGLProgram(h,[o,r],"float32")}const _y={kernelName:Ad,backendName:"webgl",kernelFunc:Py};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ly{constructor(e){this.variableNames=["X"],this.outputShape=[e,e],this.userCode=`
      void main() {
          ivec2 coords = getOutputCoords();
          float val = coords[0] == coords[1] ? getX(coords[0]) : 0.0;
          setOutput(val);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function By(n){const{inputs:e,backend:t}=n,{x:s}=e,o=[...s.shape,...s.shape],r=A(s.shape),a=F({inputs:{x:s},backend:t,attrs:{shape:[r]}}),i=new Ly(r),c=t.runWebGLProgram(i,[a],a.dtype),l=F({inputs:{x:c},backend:t,attrs:{shape:o}});return t.disposeIntermediateTensorInfo(a),t.disposeIntermediateTensorInfo(c),l}const My={kernelName:Fd,backendName:"webgl",kernelFunc:By};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Vy{constructor(e){this.variableNames=["x","W"],this.outputShape=e.outShape;const{inHeight:t,inWidth:s,padInfo:o,strideHeight:r,strideWidth:a,filterHeight:i,filterWidth:c,dilationHeight:l,dilationWidth:u}=e,{top:d,left:h}=o;this.userCode=`
      const ivec2 strides = ivec2(${r}, ${a});
      const ivec2 pads = ivec2(${d}, ${h});
      const float neg_infinity = -3.4e38;

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.w;
        ivec2 outTopLeftCorner =
            coords.yz * strides - pads;
        int hBeg = outTopLeftCorner.x;
        int wBeg = outTopLeftCorner.y;

        float curVal = neg_infinity;
        for (int h = 0; h < ${i}; h++) {
          int hIn = hBeg + h * ${l};

          if (hIn >= 0 && hIn < ${t}) {
            for (int w = 0; w < ${c}; w++) {
              int wIn = wBeg + w * ${u};

              if (wIn >= 0 && wIn < ${s}) {
                float xVal = getX(batch, hIn, wIn, d1);
                float wVal = getW(h, w, d1);

                float val = xVal + wVal;
                if (val > curVal) {
                  curVal = val;
                }
              }
            }
          }
        }

        float result = curVal;
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uy(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,filter:r}=e,{strides:a,pad:i,dilations:c}=s,l=$i(o.shape,r.shape,a,i,"NHWC",c);let u;const d=new Vy(l);u=t.runWebGLProgram(d,[o,r],"float32");const h=F({inputs:{x:u},backend:t,attrs:{shape:l.outShape}});return t.disposeIntermediateTensorInfo(u),h}const Wy={kernelName:Dd,backendName:"webgl",kernelFunc:Uy};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gy(n){const{inputs:e,backend:t,attrs:s}=n,{equation:o}=s,r=e,{allDims:a,summedDims:i,idDims:c}=sc(o,r.length);rc(a.length,c,r);const{path:l,steps:u}=ac(i,c),d=u.length;let h=null,f=a.length;const p=[];for(let x=0;x<d;++x){for(const g of u[x]){const{permutationIndices:m,expandDims:b}=oc(f,c[g]);let C;ic(m)?C=r[g]:(C=me({inputs:{x:r[g]},backend:t,attrs:{perm:m}}),p.push(C));const y=C.shape.slice();for(let S=0;S<b.length;++S)y.splice(b[S],0,1);oe(C.shape,y)||(C=F({inputs:{x:C},backend:t,attrs:{shape:y}}),p.push(C)),h===null?h=C:(h=xr({inputs:{a:C,b:h},backend:t}),p.push(h))}x<d-1&&(l[x]>=0&&(h=Ns({inputs:{x:h},backend:t,attrs:{axis:l[x]-(a.length-f),keepDims:!1}}),p.push(h)),f--)}for(const x of p)x!==h&&t.disposeIntermediateTensorInfo(x);return h}const zy={kernelName:Od,backendName:"webgl",kernelFunc:Gy};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hy="return (x >= 0.0) ? x : (exp(x) - 1.0);",Xy=`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`,qy=U({opSnippet:Hy,packedOpSnippet:Xy}),jy={kernelName:Aa,backendName:"webgl",kernelFunc:qy};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ky="return (b >= 0.0) ? a : a * (b + 1.0);",Yy=`
  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));
  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));
`,Qy=n=>{const{inputs:e,backend:t}=n,{dy:s,y:o}=e,r=$().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new cn(Yy,s.shape,o.shape):new kt(Ky,s.shape,o.shape);return t.runWebGLProgram(r,[s,o],s.dtype)},Zy={kernelName:Pd,backendName:"webgl",kernelFunc:Qy};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jy=`
  return vec4(equal(a, b));
`,e1="return float(a == b);",t1=le({opSnippet:e1,packedOpSnippet:Jy,dtype:"bool",cpuKernelImpl:A0}),n1={kernelName:Co,backendName:"webgl",kernelFunc:t1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const s1=`
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  float p = ${Qi};
  float a1 = ${Zi};
  float a2 = ${Ji};
  float a3 = ${ec};
  float a4 = ${tc};
  float a5 = ${nc};

  float sign = sign(x);
  x = abs(x);
  float t = 1.0 / (1.0 + p * x);
  return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x));
`,o1=U({opSnippet:s1}),r1={kernelName:_d,backendName:"webgl",kernelFunc:o1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const a1=ln+`
  return exp(x);
`,i1=`
  vec4 result = exp(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,gu=U({opSnippet:a1,packedOpSnippet:i1,cpuKernelImpl:F0,dtype:"float32"}),c1={kernelName:yo,backendName:"webgl",kernelFunc:gu};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fo(n){const{inputs:e,attrs:t,backend:s}=n,{dim:o}=t,{input:r}=e,a=r.shape.length,i=r.shape.slice();let c=o;return o<0&&(R(-(a+1)<=o,()=>`Axis must be in the interval [${-(a+1)}, ${a}]`),c=a+o+1),i.splice(c,0,1),F({inputs:{x:r},backend:s,attrs:{shape:i}})}const l1={kernelName:Ld,backendName:"webgl",kernelFunc:fo};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const la="return exp(x) - 1.0;",u1=U({opSnippet:la,packedOpSnippet:la,cpuKernelImpl:D0}),d1={kernelName:$o,backendName:"webgl",kernelFunc:u1};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ua{constructor(e,t,s){this.variableNames=["real","imag"];const o=t[1];this.outputShape=t;const r=s?`2.0 * ${Math.PI}`:`-2.0 * ${Math.PI}`,a=s?`${o}.0`:"1.0";let i;if(e==="real")i="return real * expR - imag * expI;";else if(e==="imag")i="return real * expI + imag * expR;";else throw new Error(`FFT component must be either "real" or "imag", got ${e}.`);this.userCode=`
      const float exponentMultiplier = ${r};

      float unaryOpComplex(float real, float expR, float imag, float expI) {
        ${i}
      }

      float mulMatDFT(int batch, int index) {
        float indexRatio = float(index) / float(${o});
        float exponentMultiplierTimesIndexRatio =
            exponentMultiplier * indexRatio;

        float result = 0.0;

        for (int i = 0; i < ${o}; i++) {
          // x = (-2|2 * PI / N) * index * i;
          float x = exponentMultiplierTimesIndexRatio * float(i);
          float expR = cos(x);
          float expI = sin(x);
          float real = getReal(batch, i);
          float imag = getImag(batch, i);

          result +=
              unaryOpComplex(real, expR, imag, expI) / ${a};
        }

        return result;
      }

      void main() {
        ivec2 coords = getOutputCoords();
        setOutput(mulMatDFT(coords[0], coords[1]));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xu(n,e,t){const s=t.texData.get(n.dataId),o=A(n.shape),r=n.shape[n.shape.length-1],a=o/r,i=F({inputs:{x:n},backend:t,attrs:{shape:[a,r]}}),c=i.shape,l=new ua("real",c,e),u=new ua("imag",c,e),d=[{dataId:s.complexTensorInfos.real.dataId,dtype:s.complexTensorInfos.real.dtype,shape:c},{dataId:s.complexTensorInfos.imag.dataId,dtype:s.complexTensorInfos.imag.dtype,shape:c}],h=t.runWebGLProgram(l,d,"float32"),f=t.runWebGLProgram(u,d,"float32"),p=ht({inputs:{real:h,imag:f},backend:t});t.disposeIntermediateTensorInfo(h),t.disposeIntermediateTensorInfo(f);const x=F({inputs:{x:p},backend:t,attrs:{shape:n.shape}});return t.disposeIntermediateTensorInfo(i),t.disposeIntermediateTensorInfo(p),x}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function h1(n){const{inputs:e,backend:t}=n,{input:s}=e;return xu(s,!1,t)}const f1={kernelName:Bd,backendName:"webgl",kernelFunc:h1};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class p1{constructor(e,t){this.outputShape=[],this.customUniforms=[{name:"value",type:"float"}],this.variableNames=["x"],this.outputShape=e,this.userCode=`
      void main() {
        // Input can be obtained from uniform value.
        setOutput(value);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zn(n){const{backend:e,attrs:t}=n,{shape:s,value:o}=t;let{dtype:r}=t;if(r=r||Yt(o),r==="string"){const a=te(r,A(s));return a.fill(o),e.makeTensorInfo(s,r,a)}else{const a=new p1(s,o),i=[[o]];return e.runWebGLProgram(a,[],r,i)}}const m1={kernelName:Fa,backendName:"webgl",kernelFunc:zn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class g1{constructor(e){this.variableNames=["Image"],this.outputShape=[];const t=e[2];this.outputShape=e,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];

          int coordX = ${t} - x - 1;
          float outputValue;
          if(coordX >= 0 && coordX < ${t}) {
            outputValue = getImage(coords[0], coords[1], coordX, coords[3]);
          } else {
            outputValue = getImage(coords[0], coords[1], coords[2], coords[3]);
          }
          setOutput(outputValue);
        }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const x1={kernelName:Md,backendName:"webgl",kernelFunc:({inputs:n,backend:e})=>{const{image:t}=n,s=e,o=new g1(t.shape);return s.runWebGLProgram(o,[t],t.dtype)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const da="return floor(x);",b1=U({opSnippet:da,packedOpSnippet:da,cpuKernelImpl:O0}),w1={kernelName:vo,backendName:"webgl",kernelFunc:b1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const C1=`
  float s = sign(a) * sign(b);
  int ia = round(a);
  int ib = round(b);
  if (ib != 0) {
    // Windows (D3D) wants guaranteed non-zero int division at compile-time.
    return float(idiv(ia, ib, s));
  } else {
    return NAN;
  }
`,y1=`
  ivec4 ia = round(a);
  ivec4 ib = round(b);
  bvec4 cond = notEqual(ib, ivec4(0));
  ivec4 result = ivec4(0);
  vec4 s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    result[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    result[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    result[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    result[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4(result);
`,$1=le({opSnippet:C1,packedOpSnippet:y1,dtype:"int32"}),v1={kernelName:xs,backendName:"webgl",kernelFunc:$1};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class S1{constructor(e){this.variableNames=["A"];const t=ge(),[s,o]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${o}.0, ${s}.0);

        vec4 values = ${t.texture2D}(A, uv);
        float value;
        if (depth == 0) {
          value = values.r;
        } else if (depth == 1) {
          value = values.g;
        } else if (depth == 2) {
          value = values.b;
        } else if (depth == 3) {
          value = values.a;
        }

        setOutput(floor(value * 255.0 + 0.5));
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class I1{constructor(e){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0;const t=ge(),[s,o]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];

        vec4 result = vec4(0.);

        for(int row=0; row<=1; row++) {
          for(int col=0; col<=1; col++) {
            texC = coords[1] + row;
            depth = coords[2] + col;

            vec2 uv = (vec2(texC, texR) + halfCR) /
                       vec2(${o}.0, ${s}.0);
            vec4 values = ${t.texture2D}(A, uv);
            float value;
            if (depth == 0) {
              value = values.r;
            } else if (depth == 1) {
              value = values.g;
            } else if (depth == 2) {
              value = values.b;
            } else if (depth == 3) {
              value = values.a;
            }

            result[row * 2 + col] = floor(value * 255.0 + 0.5);
          }
        }

        ${t.output} = result;
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const T1={kernelName:rf,backendName:"webgl",kernelFunc:R1};let Bt,Vs=$().getBool("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU");function R1(n){const{inputs:e,backend:t,attrs:s}=n;let{pixels:o}=e;const{numChannels:r}=s,a=typeof HTMLVideoElement<"u"&&o instanceof HTMLVideoElement,i=typeof HTMLImageElement<"u"&&o instanceof HTMLImageElement,[c,l]=a?[o.videoWidth,o.videoHeight]:[o.width,o.height],u=[l,c],d=[l,c,r];if(i||a){const x=$().getBool("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU");(Bt==null||x!==Vs)&&(Vs=x,Bt=document.createElement("canvas").getContext("2d",{willReadFrequently:Vs})),Bt.canvas.width=c,Bt.canvas.height=l,Bt.drawImage(o,0,0,c,l),o=Bt.canvas}const h=t.makeTensorInfo(u,"int32");t.texData.get(h.dataId).usage=ve.PIXELS,t.gpgpu.uploadPixelDataToTexture(t.getTexture(h.dataId),o);const f=$().getBool("WEBGL_PACK")?new I1(d):new S1(d),p=t.runWebGLProgram(f,[h],"int32");return t.disposeData(h.dataId),p}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function E1(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,filter:r,bias:a,preluActivationWeights:i}=e,{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:h,activation:f,leakyreluAlpha:p}=s,x=en(u),g=_e(o.shape,r.shape,c,d,l,h,!1,x);let m;const b=[],C=a!=null,y=i!=null,S=f==="leakyrelu",N=()=>{const T=[o,r],I=(v,w)=>{if(w==="NCHW"&&v.shape.length===1&&v.shape[0]!==1){const E=F({inputs:{x:v},backend:t,attrs:{shape:[v.shape[0],1,1]}});return b.push(E),E}return v};if(C&&T.push(I(a,u)),y&&T.push(I(i,u)),S){const v=t.makeTensorInfo([],"float32",Nt(p,"float32"));T.push(v),b.push(v)}return T};if(g.filterHeight===1&&g.filterWidth===1&&g.dilationHeight===1&&g.dilationWidth===1&&g.strideHeight===1&&g.strideWidth===1&&(g.padInfo.type==="SAME"||g.padInfo.type==="VALID"))m=du({x:o,filter:r,convInfo:g,backend:t,bias:a,activation:f,preluActivationWeights:i,leakyreluAlpha:p});else if(g.strideWidth<=2&&x==="channelsLast"&&$().getBool("WEBGL_EXP_CONV")){const T=f?kn(f,!0):null,I=new uu(g,C,T,y,S),v=[[g.padInfo.top,g.padInfo.left],[g.strideHeight,g.strideWidth],[g.dilationHeight,g.dilationWidth],[g.inHeight,g.inWidth]],w=N();m=t.runWebGLProgram(I,w,"float32",v)}else if($().getBool("WEBGL_CONV_IM2COL"))m=hu({x:o,filter:r,convInfo:g,backend:t,bias:a,activation:f,preluActivationWeights:i,leakyreluAlpha:p});else{const T=f?kn(f,!1):null,I=new lu(g,C,T,y,S),v=N();m=t.runWebGLProgram(I,v,"float32")}const k=F({inputs:{x:m},backend:t,attrs:{shape:g.outShape}});return b.push(m),b.forEach(T=>t.disposeIntermediateTensorInfo(T)),k}const k1={kernelName:lf,backendName:"webgl",kernelFunc:E1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function N1(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,filter:r,bias:a,preluActivationWeights:i}=e,{strides:c,pad:l,dilations:u,dimRoundingMode:d,activation:h,leakyreluAlpha:f}=s,p=[];let x=u;x==null&&(x=[1,1]),R(Jt(c,x),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${c} and dilations '${x}'`);const g=_e(o.shape,r.shape,c,x,l,d,!0),m=$().getBool("WEBGL_PACK_DEPTHWISECONV")&&g.strideWidth<=2&&g.outChannels/g.inChannels===1,b=h?kn(h,m):null,C=[o,r],y=a!=null,S=i!=null,N=h==="leakyrelu";if(y&&C.push(a),S&&C.push(i),N){const v=t.makeTensorInfo([],"float32",Nt(f,"float32"));C.push(v),p.push(v)}let k;m?k=new mu(g,y,b,S,N):k=new pu(g,y,b,S,N);const T=[[g.padInfo.top,g.padInfo.left],[g.strideHeight,g.strideWidth],[g.dilationHeight,g.dilationWidth],[g.inHeight,g.inWidth]],I=t.runWebGLProgram(k,C,"float32",T);return p.forEach(v=>t.disposeIntermediateTensorInfo(v)),I}const A1={kernelName:uf,backendName:"webgl",kernelFunc:N1};class F1{constructor(e,t,s,o){this.sliceDim=e,this.strides=t,this.paramsShape=o,this.variableNames=["x","indices"],this.outputShape=s;const r=H(s.length);let a=`
    int index;`;for(let i=0;i<this.sliceDim;i++)a+=`
          index = round(getIndices(coords[0], ${i}));
          out_of_bounds = out_of_bounds || index < 0;
          out_of_bounds = out_of_bounds || index >= ${this.paramsShape[i]};
          flattenIndex += index * ${this.strides[i]};`;this.userCode=`
         void main() {
          ${r} coords = getOutputCoords();
          int flattenIndex = 0;
          bool out_of_bounds = false;

          ${a}

          setOutput(out_of_bounds ? 0.0 : getX(flattenIndex, coords[1]));
        }
      `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function D1(n){const{inputs:e,backend:t}=n,{params:s,indices:o}=e,r=o.shape,a=r[r.length-1],i=A(s.shape),[c,l,u,d]=jo(s,o),h=F({inputs:{x:o},backend:t,attrs:{shape:[l,a]}}),f=F({inputs:{x:s},backend:t,attrs:{shape:[A(s.shape)/u,u]}});if(t.shouldExecuteOnCPU([s,o])||s.dtype==="string"){const m=t.readSync(o.dataId),b=t.bufferSync(s),C=P0(m,b,s.dtype,l,a,u,d,s.shape,i);return t.makeTensorInfo(c,s.dtype,C.values)}const p=new F1(a,d,[l,u],s.shape),x=t.runWebGLProgram(p,[f,h],f.dtype),g=F({inputs:{x},backend:t,attrs:{shape:c}});return t.disposeIntermediateTensorInfo(h),t.disposeIntermediateTensorInfo(f),t.disposeIntermediateTensorInfo(x),g}const O1={kernelName:Wd,backendName:"webgl",kernelFunc:D1};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class P1{constructor(e,t){this.variableNames=["A","indices"],this.outputShape=t,this.rank=t.length;const s=H(this.rank),o=_1(e);this.userCode=`
      void main() {
        ${s} resRC = getOutputCoords();
        int index = int(getIndices(resRC.x, resRC.z));
        float inBounds = (index >= 0) && (index < ${e[2]}) ? 1.0 : 0.0;
        setOutput(inBounds * getA(${o}));
      }
    `}}function _1(n,e){const t=["resRC.x","resRC.y","resRC.z","resRC.w"],s=[];for(let o=0;o<n.length;o++)o===2?s.push("index"):s.push(`${t[o]}`);return s.join()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bu(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,indices:r}=e,{axis:a,batchDims:i}=s,c=ce(a,o.shape)[0];if($().get("DEBUG")){const b=t.readSync(r.dataId),C=o.shape[c];for(let y=0;y<b.length;++y){const S=b[y];R(S<=C-1&&S>=0,()=>`GatherV2: the index value ${S} is not in [0, ${C-1}]`)}}const l=$c(o,r,c,i),u=A(r.shape),d=[],h=F({inputs:{x:o},backend:t,attrs:{shape:[l.batchSize,l.outerSize,l.dimSize,l.sliceSize]}}),f=F({inputs:{x:r},backend:t,attrs:{shape:[l.batchSize,u/l.batchSize]}});d.push(h),d.push(f);const p=[l.batchSize,l.outerSize,u/l.batchSize,l.sliceSize];if(t.shouldExecuteOnCPU([o,r])||o.dtype==="string"){const b=t.bufferSync(f),C=t.bufferSync(h),y=_0(C,b,p);return d.forEach(S=>t.disposeIntermediateTensorInfo(S)),t.makeTensorInfo(l.outputShape,y.dtype,y.values)}const x=new P1(h.shape,p),g=t.runWebGLProgram(x,[h,f],h.dtype);d.push(g);const m=F({inputs:{x:g},backend:t,attrs:{shape:l.outputShape}});return d.forEach(b=>t.disposeIntermediateTensorInfo(b)),m}const L1={kernelName:Ud,backendName:"webgl",kernelFunc:bu};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const B1="return float(a > b);",M1=`
  return vec4(greaterThan(a, b));
`,V1=le({opSnippet:B1,packedOpSnippet:M1,cpuKernelImpl:L0,dtype:"bool"}),U1={kernelName:So,backendName:"webgl",kernelFunc:V1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const W1="return float(a >= b);",G1=`
  return vec4(greaterThanEqual(a, b));
`,z1=le({opSnippet:W1,packedOpSnippet:G1,dtype:"bool",cpuKernelImpl:B0}),H1={kernelName:Io,backendName:"webgl",kernelFunc:z1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function X1(n){const{inputs:e,backend:t}=n,{input:s}=e;return xu(s,!0,t)}const q1={kernelName:Gd,backendName:"webgl",kernelFunc:X1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const j1="return float(!isnan(x) && !isinf(x));",K1=U({opSnippet:j1,dtype:"bool"}),Y1={kernelName:Hd,backendName:"webgl",kernelFunc:K1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Q1="return float(isinf(x));",Z1=U({opSnippet:Q1,dtype:"bool"}),J1={kernelName:Xd,backendName:"webgl",kernelFunc:Z1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const e$="return float(isnan(x));",t$=U({opSnippet:e$,dtype:"bool"}),n$={kernelName:qd,backendName:"webgl",kernelFunc:t$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const s$="return float(a < b);",o$=`
  return vec4(lessThan(a, b));
`,r$=le({opSnippet:s$,packedOpSnippet:o$,cpuKernelImpl:M0,dtype:"bool"}),a$={kernelName:To,backendName:"webgl",kernelFunc:r$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const i$="return float(a <= b);",c$=`
  return vec4(lessThanEqual(a, b));
`,l$=le({opSnippet:i$,packedOpSnippet:c$,cpuKernelImpl:V0,dtype:"bool"}),u$={kernelName:Ro,backendName:"webgl",kernelFunc:l$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function d$(n){const{backend:e,attrs:t}=n,{start:s,stop:o,num:r}=t,a=U0(s,o,r);return e.makeTensorInfo([a.length],"float32",a)}const h$={kernelName:jd,backendName:"webgl",kernelFunc:d$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const f$=ln+`
  return x < 0.0 ? 0./0. : log(x);
`,p$=`
  vec4 result = log(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : (x.r < 0.0 ? 0./0. : result.r);
  result.g = isNaN.g ? x.g : (x.g < 0.0 ? 0./0. : result.g);
  result.b = isNaN.b ? x.b : (x.b < 0.0 ? 0./0. : result.b);
  result.a = isNaN.a ? x.a : (x.a < 0.0 ? 0./0. : result.a);
  return result;
`,m$=U({opSnippet:f$,packedOpSnippet:p$,cpuKernelImpl:W0}),g$={kernelName:Eo,backendName:"webgl",kernelFunc:m$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const x$=ln+`
  return log(1.0 + x);
`,b$=U({opSnippet:x$}),w$={kernelName:Kd,backendName:"webgl",kernelFunc:b$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const C$="return float(a >= 1.0 && b >= 1.0);",y$=`
  return vec4(
    vec4(greaterThanEqual(a, vec4(1.0))) *
    vec4(greaterThanEqual(b, vec4(1.0))));
`,$$=le({opSnippet:C$,packedOpSnippet:y$,dtype:"bool"}),v$={kernelName:Yd,backendName:"webgl",kernelFunc:$$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const S$="return float(!(x >= 1.0));",I$=U({opSnippet:S$}),T$={kernelName:Qd,backendName:"webgl",kernelFunc:I$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const R$="return float(a >= 1.0 || b >= 1.0);",E$=`
  return min(
    vec4(greaterThanEqual(a, vec4(1.0))) +
    vec4(greaterThanEqual(b, vec4(1.0))),
    vec4(1.0));
`,k$=le({opSnippet:R$,packedOpSnippet:E$,dtype:"bool"}),N$={kernelName:Zd,backendName:"webgl",kernelFunc:k$};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class A${constructor(e,t,s,o,r){this.variableNames=["x"],this.outputShape=[];const a=t,i=e[3]-1;this.outputShape=e;let c;const l=`float(${s}) + float(${o}) * sum`;r===.5?c=`inversesqrt(${l})`:r===1?c=`1.0/(${l})`:c=`exp(log(${l}) * float(-${r}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];
        int d = coords[3];
        float x = getX(b, r, c, d);
        float sum = 0.0;
        for (int j = -${a}; j <= ${a}; j++) {
          int idx = d + j;
          if (idx >= 0 && idx <=  ${i}) {
            float z = getX(b, r, c, idx);
            sum += z * z;
          }
        }
        float val = x * ${c};
        setOutput(val);
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class F${constructor(e,t,s,o,r){this.variableNames=["x"],this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0;const a=t,i=e[3]-1;this.outputShape=e;let c;const l=`float(${s}) + float(${o}) * sum`;r===.5?c=`inversesqrt(${l})`:r===1?c=`1.0/(${l})`:c=`exp(log(${l}) * float(-${r}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords.x;
        int r = coords.y;
        int c = coords.z;
        int d = coords.w;

        bool hasNextCol = d < ${this.outputShape[3]};
        bool hasNextRow = c < ${this.outputShape[2]};

        vec4 sum = vec4(0.);
        vec4 xFragAtOutputCoords = getX(b, r, c, d);

        vec4 xAtOutputCoords = vec4(
          getChannel(xFragAtOutputCoords, vec2(c, d)),
          hasNextCol ?
            getChannel(xFragAtOutputCoords, vec2(c, d + 1)) : 0.0,
          hasNextRow ?
            getChannel(xFragAtOutputCoords , vec2(c + 1, d)) : 0.0,
          (hasNextRow && hasNextCol) ?
            getChannel(xFragAtOutputCoords, vec2(c + 1, d + 1)) : 0.0
        );

        int firstChannel = d - ${a};
        vec2 cache = vec2(0.);
        if(firstChannel >= 0){
          vec4 firstChannelFrag = getX(b, r, c, firstChannel);
          cache.x = getChannel(firstChannelFrag, vec2(c, firstChannel));
            if(hasNextRow){
              cache.y = getChannel(firstChannelFrag, vec2(c + 1, firstChannel));
            }
        }

        ivec2 depth = ivec2(d, d + 1);
        for (int j = - ${a}; j <= ${a}; j++) {
          ivec2 idx = depth + j;
          bvec2 aboveLowerBound = greaterThanEqual(idx, ivec2(0));
          bvec2 belowUpperBound = lessThanEqual(idx, ivec2(${i}));

          bool depthInRange = aboveLowerBound.x && belowUpperBound.x;
          bool depthPlusOneInRange = aboveLowerBound.y && belowUpperBound.y;

          if(depthInRange || depthPlusOneInRange){
            vec4 z = vec4(0.);
            vec4 xFragAtCurrentDepth;
            z.xz = cache.xy;
            if(depthPlusOneInRange && hasNextCol){
              xFragAtCurrentDepth = idx.y != d ?
                getX(b, r, c, idx.y) : xFragAtOutputCoords;
              z.y = getChannel(xFragAtCurrentDepth, vec2(c, idx.y));
              if(hasNextRow){
                z.w = getChannel(xFragAtCurrentDepth, vec2(c + 1, idx.y));
              }
            }
            cache.xy = z.yw;
            sum += z * z;
          }
        }
        vec4 result = xAtOutputCoords * ${c};
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const D$=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{depthRadius:r,bias:a,alpha:i,beta:c}=s,l=$().getBool("WEBGL_PACK_NORMALIZATION")?new F$(o.shape,r,a,i,c):new A$(o.shape,r,a,i,c);return t.runWebGLProgram(l,[o],o.dtype)},O$={kernelName:Jd,backendName:"webgl",kernelFunc:D$};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class P${constructor(e,t,s,o,r){this.variableNames=["inputImage","outputImage","dy"],this.outputShape=[],this.outputShape=e,this.depth=e[3],this.depthRadius=t,this.bias=s,this.alpha=o,this.beta=r,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];

        float result = 0.0;
        for (int d = 0; d < ${this.depth}; ++d) {
          int depthBegin = int(max(0.0, float(d - ${t})));
          int depthEnd = int(min(float(${this.depth}),
              float(d + ${t} + 1)));

          const int MIN_DEPTH_BEGIN = 0;
          const int MAX_DEPTH_END = ${this.depth};

          float norm = 0.0;
          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd) {
              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);
            }
            else {
              break;
            }
          }

          norm = float(${o}) * norm + float(${s});

          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd){
              float dyi = -2.0 * float(${o})
                * float(${r})
                * getInputImage(b, r, c, k) * getOutputImage(b, r, c, d)
                / norm;
              if (k == d) {
                dyi += pow(norm, -1.0 * ${r});
              }
              if (k == coords[3]) {
                dyi *= getDy(b, r, c, d);
                result += dyi;
              }
            }
            else {
              break;
            }
          }
      }
      setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _$=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:o,y:r,dy:a}=e,{depthRadius:i,bias:c,alpha:l,beta:u}=s,d=new P$(o.shape,i,c,l,u);return t.runWebGLProgram(d,[o,r,a],o.dtype)},L$={kernelName:eh,backendName:"webgl",kernelFunc:_$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function B$(n,e,t,s){const o=A(e),a=A(n.shape)/o,i=F({inputs:{x:n},attrs:{shape:[a,o]},backend:s}),c=_t(i,n.dtype,"max",s),l=F({inputs:{x:c},attrs:{shape:t},backend:s});return s.disposeIntermediateTensorInfo(i),s.disposeIntermediateTensorInfo(c),l}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wu(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{reductionIndices:r,keepDims:a}=s,i=o.shape.length,c=ce(r,o.shape);let l=c;const u=Re(l,i),d=u!=null,h=t.shouldExecuteOnCPU([o]);let f=o;if(d){if(h){const C=t.texData.get(f.dataId).values,y=new Array(i);for(let k=0;k<y.length;k++)y[k]=o.shape[u[k]];const S=mr(C,o.shape,o.dtype,u,y);f=t.makeTensorInfo(y,o.dtype);const N=t.texData.get(f.dataId);N.values=S}else f=ks(o,u,t);l=Ee(l.length,i)}Le("max",l,i);const[p,x]=He(f.shape,l);let g=p;a&&(g=Xe(p,c));let m;if(h){const C=t.texData.get(f.dataId).values,y=G0(C,A(x),g,o.dtype);m=t.makeTensorInfo(g,o.dtype);const S=t.texData.get(m.dataId);S.values=y}else m=B$(f,x,g,t);return d&&t.disposeIntermediateTensorInfo(f),m}const M$={kernelName:th,backendName:"webgl",kernelFunc:wu};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const V$=gr+`
  return max(a, b);
`,U$=`
  vec4 result = vec4(max(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Pt+`
  return result;
`,W$=le({opSnippet:V$,packedOpSnippet:U$,cpuKernelImpl:z0}),G$={kernelName:ws,backendName:"webgl",kernelFunc:W$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function z$(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e;nn(o,"maxPool");const{filterSize:r,strides:a,pad:i,dimRoundingMode:c}=s,l=1;R(Jt(a,l),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${a} and dilations '${l}'`);const u=Zt(o.shape,r,a,l,i,c);if(u.filterWidth===1&&u.filterHeight===1&&oe(u.inShape,u.outShape))return ye({inputs:{x:o},backend:t});const d=new Nn(u,"max",!1);return t.runWebGLProgram(d,[o],o.dtype)}const H$={kernelName:nh,backendName:"webgl",kernelFunc:z$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function X$(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{filterSize:r,strides:a,pad:i,dataFormat:c,dimRoundingMode:l}=s,u=[1,1,1],d=Ln(o.shape,r,a,u,i,l,c),h=new br(d,"max",!1);return t.runWebGLProgram(h,[o],o.dtype)}const q$={kernelName:oh,backendName:"webgl",kernelFunc:X$};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class j${constructor(e){this.variableNames=["dy","maxPos"],this.outputShape=e.inShape;const t=e.strideHeight,s=e.strideWidth,o=e.dilationHeight,r=e.effectiveFilterHeight,a=e.effectiveFilterWidth,i=r-1-e.padInfo.top,c=a-1-e.padInfo.left,l=r*a-1;this.userCode=`
      const ivec2 pads = ivec2(${i}, ${c});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${r};
          wR += ${o}) {
          float dyR = float(dyRCorner + wR) / ${t}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${a}; wC++) {
            float dyC = float(dyCCorner + wC) / ${s}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);
            int maxPosValue = ${l} - int(getMaxPos(b, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            int curPosValue = wR * ${a} + wC;
            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

            dotProd += dyValue * mask;
          }
        }
        setOutput(dotProd);
      }
    `}}class K${constructor(e){this.variableNames=["dy","maxPos"],this.outputShape=e.inShape;const t=e.strideDepth,s=e.strideHeight,o=e.strideWidth,r=e.dilationDepth,a=e.dilationHeight,i=e.dilationWidth,c=e.effectiveFilterDepth,l=e.effectiveFilterHeight,u=e.effectiveFilterWidth,d=c-1-e.padInfo.front,h=l-1-e.padInfo.top,f=u-1-e.padInfo.left,p=c*l*u-1;this.userCode=`
      const ivec3 pads = ivec3(${d}, ${h}, ${f});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${c};
           wD += ${r}) {
          float dyD = float(dyDCorner + wD) / ${t}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${l};
              wR += ${a}) {
            float dyR = float(dyRCorner + wR) / ${s}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${u};
                wC += ${i}) {
              float dyC = float(dyCCorner + wC) / ${o}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);
              int maxPosValue = ${p} -
                  int(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              int curPosValue =
                  wD * ${l} * ${u} +
                  wR * ${u} + wC;
              float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

              dotProd += dyValue * mask;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Y$(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,input:r}=e,a=r,{filterSize:i,strides:c,pad:l,dimRoundingMode:u}=s,d=[1,1,1],h=Ln(a.shape,i,c,d,l,u),f=new br(h,"max",!0),p=t.runWebGLProgram(f,[a],a.dtype),x=new K$(h),g=t.runWebGLProgram(x,[o,p],a.dtype);return t.disposeIntermediateTensorInfo(p),g}const Q$={kernelName:rh,backendName:"webgl",kernelFunc:Y$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Z$(n){const{inputs:e,backend:t,attrs:s}=n,{dy:o,input:r,output:a}=e,i=r;nn([r,a],"maxPoolGrad");const{filterSize:c,strides:l,pad:u,dimRoundingMode:d}=s,h=Zt(i.shape,c,l,1,u,d),f=!0,p=new Nn(h,"max",f),x=t.runWebGLProgram(p,[i],i.dtype),g=new j$(h),m=t.runWebGLProgram(g,[o,x],i.dtype);return t.disposeIntermediateTensorInfo(x),m}const J$={kernelName:sh,backendName:"webgl",kernelFunc:Z$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ev(n,e,t,s){let o=new Nn(t,"max",!1);const r=s.runWebGLProgram(o,[n],"float32");o=new Nn(t,"max",!0,!0,e);const a=s.runWebGLProgram(o,[n],"float32");return[r,a]}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tv={kernelName:ah,backendName:"webgl",kernelFunc:({inputs:n,attrs:e,backend:t})=>{const{x:s}=n,{filterSize:o,strides:r,pad:a,includeBatchInIndex:i}=e,c=t;R(s.shape.length===4,()=>`Error in maxPool: input must be rank 4 but got rank ${s.shape.length}.`);const l=[1,1];R(Jt(r,l),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${r} and dilations '${l}'`);const u=Zt(s.shape,o,r,l,a),[d,h]=ev(s,i,u,c);return[d,h]}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nv(n,e,t,s){const o=A(e),a=A(n.shape)/o,i=F({inputs:{x:n},attrs:{shape:[a,o]},backend:s}),c=_t(i,"float32","mean",s),l=F({inputs:{x:c},attrs:{shape:t},backend:s});return s.disposeIntermediateTensorInfo(i),s.disposeIntermediateTensorInfo(c),l}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sv={kernelName:ih,backendName:"webgl",kernelFunc:({inputs:n,attrs:e,backend:t})=>{const{x:s}=n,{keepDims:o,axis:r}=e,a=t,i=s.shape.length,c=ce(r,s.shape);let l=c;const u=Re(l,i),d=u!=null,h=a.shouldExecuteOnCPU([s]),f=[];let p=s;if(d){if(h){const y=a.texData.get(p.dataId).values,S=new Array(i);for(let T=0;T<S.length;T++)S[T]=s.shape[u[T]];const N=mr(y,s.shape,s.dtype,u,S);p=a.makeTensorInfo(S,s.dtype);const k=a.texData.get(p.dataId);k.values=N}else p=ks(s,u,a);f.push(p),l=Ee(l.length,i)}Le("sum",l,i);const[x,g]=He(p.shape,l);let m=x;o&&(m=Xe(x,c));const b=nv(p,g,m,a);for(const C of f)a.disposeIntermediateTensorInfo(C);return b}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ov(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,keepDims:a}=s,i=o.shape.length,c=ce(r,o.shape);let l=c;const u=Re(l,i);let d=o;u!=null&&(d=me({inputs:{x:o},backend:t,attrs:{perm:u}}),l=Ee(l.length,o.shape.length)),Le("min",l,i);const[h,f]=He(d.shape,l),p=A(f),x=F({inputs:{x:d},backend:t,attrs:{shape:[-1,p]}}),g=_t(x,x.dtype,"min",t);let m;if(a){const b=Xe(h,c);m=F({inputs:{x:g},backend:t,attrs:{shape:b}})}else m=F({inputs:{x:g},backend:t,attrs:{shape:h}});return t.disposeIntermediateTensorInfo(x),t.disposeIntermediateTensorInfo(g),u!=null&&t.disposeIntermediateTensorInfo(d),m}const rv={kernelName:ch,backendName:"webgl",kernelFunc:ov};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const av=gr+`
  return min(a, b);
`,iv=`
  vec4 result = vec4(min(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Pt+`
  return result;
`,cv=le({opSnippet:av,packedOpSnippet:iv,cpuKernelImpl:H0}),lv={kernelName:ko,backendName:"webgl",kernelFunc:cv};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class uv{constructor(e,t,s){this.variableNames=["x"],this.outputShape=t.map((u,d)=>u[0]+e[d]+u[1]);const o=e.length,r=H(o),a=t.map(u=>u[0]).join(","),i=t.map((u,d)=>u[0]+e[d]).join(","),c=["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,o),l=s==="reflect"?0:1;if(o===1){this.userCode=`
        int start = ${a};
        int end = ${i};

        void main() {
          int outC = getOutputCoords();
          if (outC < start) {
            outC = start * 2 - outC - ${l};
          } else if(outC >= end) {
            outC = (end - 1) * 2 - outC + ${l};
          }
          setOutput(getX(outC - start));
        }
      `;return}this.userCode=`
      ${r} start = ${r}(${a});
      ${r} end = ${r}(${i});

      void main() {
        ${r} outC = getOutputCoords();
        for (int i = 0; i < ${o}; i++) {
          if (outC[i] < start[i]) {
            outC[i] = start[i] * 2 - outC[i] - ${l};
          } else if(outC[i] >= end[i]) {
            outC[i] = (end[i] - 1) * 2 - outC[i] + ${l};
          }
        }
        ${r} coords = outC - start;
        setOutput(getX(${c}));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class dv{constructor(e,t,s){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t.map((p,x)=>p[0]+e[x]+p[1]);const o=e.length,r=H(o),a=t.map(p=>p[0]).join(","),i=t.map((p,x)=>p[0]+e[x]).join(","),c=pe("rc",o),l=pe("source",o),u=`${c[o-1]} < ${this.outputShape[o-1]}`,d=o===1?"source":`vec2(${l.slice(-2).join()})`,h=s==="reflect"?0:1;let f="";if(o===1){const p=`
        ${r} source = rc;
        if (source < start) {
          source = start * 2 - source - ${h};
        } else if (source >= end) {
          source = (end - 1) * 2 - source + ${h};
        }
        source -= start;
      `;f=`
        ${r} rc = outputLoc;
        ${p}
        result[0] = getChannel(getX(${l.join()}), ${d});
        ${c[o-1]} += 1;
        if(${u}) {
          ${p}
          result[1] = getChannel(getX(${l.join()}), ${d});
        }
      `}else{const p=`
        ${r} source = rc;
        ${r} lt = ${r}(lessThan(source, start));
        ${r} gte = ${r}(greaterThanEqual(source, end));
        ${r} orig = 1 - (lt + gte);
        source = orig * source +
                lt * (start * 2 - source - ${h}) +
                gte * ((end - 1) * 2 - source + ${h});
        source -= start;
      `;f=`
        ${r} rc = outputLoc;
        ${p}
        result[0] = getChannel(getX(${l.join()}), ${d});
        ${c[o-1]} += 1;
        if(${u}) {
          ${p}
          result[1] = getChannel(getX(${l.join()}), ${d});
        }
        rc = outputLoc;
        ${c[o-2]} += 1;
        if(${c[o-2]} < ${this.outputShape[o-2]}) {
          ${p}
          result[2] = getChannel(getX(${l.join()}), ${d});
          ${c[o-1]} += 1;
          if(${u}) {
            ${p}
            result[3] = getChannel(getX(${l.join()}), ${d});
          }
        }
      `}this.userCode=`
      const ${r} start = ${r}(${a});
      const ${r} end = ${r}(${i});

      void main() {
        ${r} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${f}
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hv=({inputs:n,backend:e,attrs:t})=>{const{x:s}=n,{paddings:o,mode:r}=t,a=$().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new dv(s.shape,o,r):new uv(s.shape,o,r);return e.runWebGLProgram(a,[s],s.dtype)},fv={kernelName:lh,backendName:"webgl",kernelFunc:hv};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pv=`if (b == 0.0) return NAN;
  return mod(a, b);`,mv=`
  vec4 result = mod(a, b);
  bvec4 isNaN = equal(b, vec4(0.0));
  `+Pt+`
  return result;
`,gv=le({opSnippet:pv,packedOpSnippet:mv}),xv={kernelName:uh,backendName:"webgl",kernelFunc:gv};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class bv{constructor(e,t,s){this.variableNames=["probs"],this.customUniforms=[{name:"seed",type:"float"}],this.outputShape=[e,s],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];

        float r = random(seed);
        float cdf = 0.0;

        for (int i = 0; i < ${t-1}; i++) {
          cdf += getProbs(batch, i);

          if (r < cdf) {
            setOutput(float(i));
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutput(float(${t-1}));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wv=`
if (a == b) {
  return 1.0;
};
return a / b;`,Cv=`
  // vec4 one = vec4(equal(a, b));
  // return one + (vec4(1.0) - one) * a / b;
  vec4 result = a / b;
  if(a.x == b.x) {
    result.x = 1.;
  }
  if(a.y == b.y) {
    result.y = 1.;
  }
  if(a.z == b.z) {
    result.z = 1.;
  }
  if(a.w == b.w) {
    result.w = 1.;
  }

  return result;
`,Cu=le({opSnippet:wv,packedOpSnippet:Cv,checkOutOfBounds:!0}),yv={kernelName:Na,backendName:"webgl",kernelFunc:Cu};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ha="return a - b;",yu=le({opSnippet:ha,packedOpSnippet:ha,supportsComplex:!0,cpuKernelImpl:hb}),$v={kernelName:vs,backendName:"webgl",kernelFunc:yu};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $u(n){const{inputs:e,backend:t,attrs:s}=n,{logits:o}=e,{dim:r}=s,a=ce([r],o.shape),i=wu({inputs:{x:o},backend:t,attrs:{reductionIndices:a,keepDims:!1}}),c=Xe(i.shape,a),l=F({inputs:{x:i},backend:t,attrs:{shape:c}}),u=yu({inputs:{a:o,b:l},backend:t}),d=gu({inputs:{x:u},backend:t}),h=Ns({inputs:{x:d},backend:t,attrs:{axis:a,keepDims:!1}}),f=F({inputs:{x:h},backend:t,attrs:{shape:c}}),p=Cu({inputs:{a:d,b:f},backend:t});return t.disposeIntermediateTensorInfo(i),t.disposeIntermediateTensorInfo(l),t.disposeIntermediateTensorInfo(u),t.disposeIntermediateTensorInfo(d),t.disposeIntermediateTensorInfo(h),t.disposeIntermediateTensorInfo(f),p}const vv={kernelName:Uh,backendName:"webgl",kernelFunc:$u};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sv(n){const{inputs:e,backend:t,attrs:s}=n,{logits:o}=e,{numSamples:r,seed:a,normalized:i}=s,c=i?o:$u({inputs:{logits:o},backend:t,attrs:{dim:o.shape.length-1}}),l=c.shape[0],u=c.shape[1],d=new bv(l,u,r),h=[[a]],f=t.runWebGLProgram(d,[c],"int32",h);return i||t.disposeIntermediateTensorInfo(c),f}const Iv={kernelName:dh,backendName:"webgl",kernelFunc:Sv};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tv=Fe+`
  return -x;
`,Rv=`
  vec4 result = -x;
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`;function Ev(n){const{inputs:e,backend:t}=n,{x:s}=e;if(t.shouldExecuteOnCPU([s])){const r=t.texData.get(s.dataId),[a,i]=q0(r.values,s.shape,s.dtype);return t.makeTensorInfo(i,s.dtype,a)}let o;return $().getBool("WEBGL_PACK_UNARY_OPERATIONS")?o=new nt(s.shape,Rv):o=new Be(s.shape,Tv),t.runWebGLProgram(o,[s],s.dtype)}const kv={kernelName:Oa,backendName:"webgl",kernelFunc:Ev};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nv=gm;function Av(n){Ae("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:e,backend:t,attrs:s}=n,{boxes:o,scores:r}=e,{maxOutputSize:a,iouThreshold:i,scoreThreshold:c}=s,l=t.readSync(o.dataId),u=t.readSync(r.dataId),{selectedIndices:d}=Nv(l,u,a,i,c);return t.makeTensorInfo([d.length],"int32",new Int32Array(d))}const Fv={kernelName:hh,backendName:"webgl",kernelFunc:Av};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Dv=xm;function Ov(n){Ae("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:e,backend:t,attrs:s}=n,{boxes:o,scores:r}=e,{maxOutputSize:a,iouThreshold:i,scoreThreshold:c,padToMaxOutputSize:l}=s,u=t.readSync(o.dataId),d=t.readSync(r.dataId),{selectedIndices:h,validOutputs:f}=Dv(u,d,a,i,c,l);return[t.makeTensorInfo([h.length],"int32",new Int32Array(h)),t.makeTensorInfo([],"int32",new Int32Array([f]))]}const Pv={kernelName:fh,backendName:"webgl",kernelFunc:Ov};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _v=bm;function Lv(n){Ae("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:e,backend:t,attrs:s}=n,{boxes:o,scores:r}=e,{maxOutputSize:a,iouThreshold:i,scoreThreshold:c,softNmsSigma:l}=s,u=t.readSync(o.dataId),d=t.readSync(r.dataId),h=a,f=i,p=c,x=l,{selectedIndices:g,selectedScores:m}=_v(u,d,h,f,p,x);return[t.makeTensorInfo([g.length],"int32",new Int32Array(g)),t.makeTensorInfo([m.length],"float32",new Float32Array(m))]}const Bv={kernelName:ph,backendName:"webgl",kernelFunc:Lv};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Mv{constructor(e,t,s,o){this.variableNames=["indices"],this.outputShape=[e,t],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int index = round(getIndices(coords.x));
        setOutput(mix(float(${o}), float(${s}),
                      float(index == coords.y)));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Vv=n=>{const{inputs:e,backend:t,attrs:s}=n,{indices:o}=e,{dtype:r,depth:a,onValue:i,offValue:c}=s,l=A(o.shape),u=new Mv(l,a,i,c),d=F({inputs:{x:o},backend:t,attrs:{shape:[l]}}),h=t.runWebGLProgram(u,[d],r);t.disposeIntermediateTensorInfo(d);const f=[...o.shape,a],p=F({inputs:{x:h},backend:t,attrs:{shape:f}});return t.disposeIntermediateTensorInfo(h),p},Uv={kernelName:gh,backendName:"webgl",kernelFunc:Vv};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fs(n){const{inputs:e,backend:t}=n,{x:s}=e;if(s.dtype==="complex64"){const o=Gn({inputs:{input:s},backend:t}),r=fs({inputs:{x:o},backend:t}),a=As({inputs:{input:s},backend:t}),i=fs({inputs:{x:a},backend:t}),c=ht({inputs:{real:r,imag:i},backend:t});return t.disposeIntermediateTensorInfo(o),t.disposeIntermediateTensorInfo(r),t.disposeIntermediateTensorInfo(a),t.disposeIntermediateTensorInfo(i),c}else return zn({attrs:{shape:s.shape,dtype:s.dtype,value:s.dtype==="string"?"":0},backend:t})}const Wv={kernelName:Xa,backendName:"webgl",kernelFunc:fs};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vu(n){const{inputs:e,backend:t}=n,{x:s}=e;if(s.dtype==="string")throw new Error("onesLike is not supported under string dtype");if(s.dtype==="complex64"){const o=Gn({inputs:{input:s},backend:t}),r=vu({inputs:{x:o},backend:t}),a=As({inputs:{input:s},backend:t}),i=fs({inputs:{x:a},backend:t}),c=ht({inputs:{real:r,imag:i},backend:t});return t.disposeIntermediateTensorInfo(o),t.disposeIntermediateTensorInfo(r),t.disposeIntermediateTensorInfo(a),t.disposeIntermediateTensorInfo(i),c}else return zn({attrs:{shape:s.shape,dtype:s.dtype,value:1},backend:t})}const Gv={kernelName:mh,backendName:"webgl",kernelFunc:vu};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zv(n){const{inputs:e,backend:t,attrs:s}=n,{axis:o}=s;if(e.length===1)return fo({inputs:{input:e[0]},backend:t,attrs:{dim:o}});const r=e[0].shape,a=e[0].dtype;e.forEach(u=>{Kt(r,u.shape,"All tensors passed to stack must have matching shapes"),R(a===u.dtype,()=>"All tensors passed to stack must have matching dtypes")});const i=[],c=e.map(u=>{const d=fo({inputs:{input:u},backend:t,attrs:{dim:o}});return i.push(d),d}),l=cu({inputs:c,backend:t,attrs:{axis:o}});return i.forEach(u=>t.disposeIntermediateTensorInfo(u)),l}const Hv={kernelName:xh,backendName:"webgl",kernelFunc:zv};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Xv{constructor(e,t,s){this.variableNames=["x"],this.customUniforms=[{name:"value",type:"float"}],this.outputShape=t.map((l,u)=>l[0]+e[u]+l[1]);const o=e.length,r=H(o),a=t.map(l=>l[0]).join(","),i=t.map((l,u)=>l[0]+e[u]).join(","),c=["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,o);if(o===1){this.userCode=`
        int start = ${a};
        int end = ${i};

        void main() {
          int outC = getOutputCoords();
          if (outC < start || outC >= end) {
            setOutput(value);
          } else {
            setOutput(getX(outC - start));
          }
        }
      `;return}this.userCode=`
      ${r} start = ${r}(${a});
      ${r} end = ${r}(${i});

      void main() {
        ${r} outC = getOutputCoords();
        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {
          setOutput(value);
        } else {
          ${r} coords = outC - start;
          setOutput(getX(${c}));
        }
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class qv{constructor(e,t,s){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"value",type:"float"}],this.outputShape=t.map((x,g)=>x[0]+e[g]+x[1]);const o=e.length,r=H(o),a=t.map(x=>x[0]).join(","),i=t.map((x,g)=>x[0]+e[g]).join(","),c=pe("rc",o),l=pe("source",o),u=`${c[o-1]} < ${this.outputShape[o-1]}`,d=o===1?"source":`vec2(${l.slice(-2).join()})`,h=[`${r} rc = outputLoc;`,`${c[o-1]} += 1;
       if(${u}) {
      `,o===1?"":`}
       rc = outputLoc;
       ${c[o-2]} += 1;
       if(${c[o-2]} < ${this.outputShape[o-2]}) {`,o===1?"":`  ${c[o-1]} += 1;
         if(${u}) {`],f=o===1?"rc < start || rc >= end":"any(lessThan(rc, start)) || any(greaterThanEqual(rc, end))";let p="";for(let x=0,g=o===1?2:4;x<g;x++)p+=`
        ${h[x]}
        if (${f}) {
          result[${x}] = float(value);
        } else {
          ${r} source = rc - start;
          result[${x}] = getChannel(getX(${l.join()}), ${d});
        }
      `;p+=o===1?"} ":"}}",this.userCode=`
      const ${r} start = ${r}(${a});
      const ${r} end = ${r}(${i});

      void main() {
        ${r} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${p}
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Su=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{paddings:r,constantValue:a}=s;if(A(o.shape)===0){const l=r.map((u,d)=>u[0]+o.shape[d]+u[1]);return zn({backend:t,attrs:{shape:l,value:a,dtype:o.dtype}})}const i=$().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new qv(o.shape,r,a):new Xv(o.shape,r,a),c=[[a]];return t.runWebGLProgram(i,[o],o.dtype,c)},jv={kernelName:bh,backendName:"webgl",kernelFunc:Su};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kv=`
  if(a < 0.0 && floor(b) < b){
    return NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  return (round(mod(b, 2.0)) != 1) ?
      pow(abs(a), b) : sign(a) * pow(abs(a), b);
`,Yv=`
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  bvec4 isExpZero = equal(b, vec4(0.0));
  result.r = isExpZero.r ? 1.0 : result.r;
  result.g = isExpZero.g ? 1.0 : result.g;
  result.b = isExpZero.b ? 1.0 : result.b;
  result.a = isExpZero.a ? 1.0 : result.a;

  bvec4 isNaN1 = lessThan(a, vec4(0.0));
  bvec4 isNaN2 = lessThan(floor(b), b);
  bvec4 isNaN = bvec4(isNaN1.x && isNaN2.x, isNaN1.y && isNaN2.y, isNaN1.z && isNaN2.z, isNaN1.w && isNaN2.w);
  `+Pt+`
  return result;
`,Qv=le({opSnippet:Kv,packedOpSnippet:Yv}),Zv={kernelName:Pa,backendName:"webgl",kernelFunc:Qv};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jv(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:r,keepDims:a}=s,i=o.shape.length,c=[],l=ce(r,o.shape);let u=l;const d=Re(u,i);let h=o;d!=null&&(h=me({inputs:{x:o},backend:t,attrs:{perm:d}}),u=Ee(u.length,i),c.push(h)),Le("prod",u,i);let f;if(t.shouldExecuteOnCPU([h])){const p=t.texData.get(h.dataId).values,{outVals:x,outShape:g,outDtype:m}=K0(h.shape,h.dtype,p,u);f=t.makeTensorInfo(g,m,x)}else{const[p,x]=He(h.shape,u),g=A(x),m=F({inputs:{x:h},backend:t,attrs:{shape:[-1,g]}}),b=Po(o.dtype),C=_t(m,b,"prod",t);f=F({inputs:{x:C},backend:t,attrs:{shape:p}}),c.push(m),c.push(C)}if(a){c.push(f);const p=Xe(f.shape,l);f=F({inputs:{x:f},backend:t,attrs:{shape:p}})}return c.forEach(p=>t.disposeIntermediateTensorInfo(p)),f}const eS={kernelName:La,backendName:"webgl",kernelFunc:Jv};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tS(n){const{inputs:e,backend:t,attrs:s}=n,{paramsNestedSplits:o,paramsDenseValues:r,indices:a}=e,{outputRaggedRank:i}=s,c=o.map(m=>t.readSync(m.dataId)),l=o.map(m=>m.shape),u=t.readSync(r.dataId),d=t.readSync(a.dataId),[h,f,p]=Y0(c,l,u,r.shape,r.dtype,d,a.shape,i),x=h.map(m=>t.makeTensorInfo([m.length],"int32",m)),g=t.makeTensorInfo(p,r.dtype,f);return x.concat([g])}const nS={kernelName:wh,backendName:"webgl",kernelFunc:tS};/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sS(n){const{inputs:e,backend:t}=n,{starts:s,limits:o,deltas:r}=e,a=t.readSync(s.dataId),i=t.readSync(o.dataId),c=t.readSync(r.dataId),[l,u]=Q0(a,s.shape,s.dtype,i,o.shape,c,r.shape),d=t.makeTensorInfo([l.length],"int32",l),h=t.makeTensorInfo([u.length],s.dtype,u);return[d,h]}const oS={kernelName:Ch,backendName:"webgl",kernelFunc:sS};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rS(n){const{inputs:e,backend:t,attrs:s}=n,{shape:o,values:r,defaultValue:a,rowPartitionTensors:i}=e,{rowPartitionTypes:c}=s,l=t.readSync(o.dataId),u=t.readSync(r.dataId),d=t.readSync(a.dataId),h=i.map(g=>t.readSync(g.dataId)),f=i.map(g=>g.shape),[p,x]=Z0(l,o.shape,u,r.shape,r.dtype,d,a.shape,h,f,c);return t.makeTensorInfo(p,r.dtype,x)}const aS={kernelName:yh,backendName:"webgl",kernelFunc:rS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Iu=n=>{const{backend:e,attrs:t}=n,{start:s,stop:o,step:r,dtype:a}=t,i=J0(s,o,r,a);return e.makeTensorInfo([i.length],a,i)},iS={kernelName:$h,backendName:"webgl",kernelFunc:Iu};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cS="return 1.0 / x;",lS=U({opSnippet:cS}),uS={kernelName:vh,backendName:"webgl",kernelFunc:lS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dS=Fe+`
  return (x < 0.0) ? 0.0 : x;
`,hS=`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,fS=U({opSnippet:dS,packedOpSnippet:hS}),pS={kernelName:Ma,backendName:"webgl",kernelFunc:fS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mS=Fe+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,gS=`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,xS=U({opSnippet:mS,packedOpSnippet:gS}),bS={kernelName:Ua,backendName:"webgl",kernelFunc:xS};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class wS{constructor(e,t,s,o,r){this.variableNames=["A"],this.outputShape=[];const[a,i,c,l]=e;this.outputShape=[a,t,s,l];const u=[o&&t>1?i-1:i,o&&s>1?c-1:c],d=[o&&t>1?t-1:t,o&&s>1?s-1:s];let h;r?h="(vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC - vec2(0.5)":h="vec2(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${u[0]/d[0]},
          ${u[1]/d[1]});
      const vec2 inputShapeRC = vec2(${i}.0, ${c}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${h};

        // Compute the four integer indices.
        ivec2 sourceFloorRC = ivec2(max(sourceFracIndexRC, vec2(0.0)));
        ivec2 sourceCeilRC = ivec2(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        float topLeft = getA(b, sourceFloorRC.x, sourceFloorRC.y, d);
        float bottomLeft = getA(b, sourceCeilRC.x, sourceFloorRC.y, d);
        float topRight = getA(b, sourceFloorRC.x, sourceCeilRC.y, d);
        float bottomRight = getA(b, sourceCeilRC.x, sourceCeilRC.y, d);

        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

        float top = topLeft + (topRight - topLeft) * fracRC.y;
        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
        float newValue = top + (bottom - top) * fracRC.x;

        setOutput(newValue);
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class CS{constructor(e,t,s,o,r){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];const[a,i,c,l]=e;this.outputShape=[a,t,s,l];const u=[o&&t>1?i-1:i,o&&s>1?c-1:c],d=[o&&t>1?t-1:t,o&&s>1?s-1:s];let h;r?h="(vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC - vec3(0.5)":h="vec3(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${u[0]/d[0]},
          ${u[1]/d[1]},
          ${u[1]/d[1]});
      const vec3 inputShapeRC = vec3(${i}.0, ${c}.0,
                                     ${c}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${h};

        // Compute the four integer indices.
        ivec3 sourceFloorRC = ivec3(max(sourceFracIndexRC, vec3(0.0)));
        ivec3 sourceCeilRC = ivec3(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${l-1};
        bool hasNextRow = coords.z < ${s-1};

        // In parallel, construct four corners for all four components in
        // packed 2x2 cell.
        vec4 topLeft = vec4(
          getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 bottomLeft = vec4(
          getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 topRight = vec4(
          getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec4 bottomRight = vec4(
          getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec3 fracRC = sourceFracIndexRC - vec3(sourceFloorRC);

        vec4 top = mix(topLeft, topRight, fracRC.yyzz);
        vec4 bottom = mix(bottomLeft, bottomRight, fracRC.yyzz);
        vec4 newValue = mix(top, bottom, fracRC.x);

        setOutput(newValue);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yS(n){const{inputs:e,backend:t,attrs:s}=n,{images:o}=e,{alignCorners:r,halfPixelCenters:a,size:i}=s,[c,l]=i,u=$().getBool("WEBGL_PACK_IMAGE_OPERATIONS")?new CS(o.shape,c,l,r,a):new wS(o.shape,c,l,r,a);return t.runWebGLProgram(u,[o],"float32")}const $S={kernelName:Th,backendName:"webgl",kernelFunc:yS};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class vS{constructor(e,t,s){this.variableNames=["dy"],this.outputShape=[],this.outputShape=t;const[,o,r]=t,[,a,i]=e,c=[s&&a>1?o-1:o,s&&i>1?r-1:r],l=[s&&a>1?a-1:a,s&&i>1?i-1:i],u=c[0]/l[0],d=c[1]/l[1],h=1/u,f=1/d,p=Math.ceil(h)*2+2,x=Math.ceil(f)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${u});
        const float widthScale = float(${d});

        const float invHeightScale = float(${h});
        const float invWidthScale = float(${f});

        const int winHeight = int(${p});
        const int winWidth = int(${x});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(startRLerp - float(winHeight / 2));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(startCLerp - float(winWidth / 2));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${a}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${i}) {
              continue;
            }

            float dxR = float(dyR) * heightScale;
            int topDxRIndex = int(floor(dxR));
            int bottomDxRIndex = int(min(ceil(dxR), ${o-1}.0));
            float dxRLerp = dxR - float(topDxRIndex);
            float inverseDxRLerp = 1.0 - dxRLerp;

            float dxC = float(dyC) * widthScale;
            int leftDxCIndex = int(floor(dxC));
            int rightDxCIndex = int(min(ceil(dxC), ${r-1}.0));
            float dxCLerp = dxC - float(leftDxCIndex);
            float inverseDxCLerp = 1.0 - dxCLerp;

            if (r == topDxRIndex && c == leftDxCIndex) {
              // topLeft
              accumulator +=
                getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;
            }

            if (r == topDxRIndex && c == rightDxCIndex) {
              // topRight
              accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;
            }

            if (r == bottomDxRIndex && c == leftDxCIndex) {
              // bottomLeft
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;
            }

            if (r == bottomDxRIndex && c == rightDxCIndex) {
              // bottomRight
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function SS(n){const{inputs:e,backend:t,attrs:s}=n,{images:o,dy:r}=e,{alignCorners:a}=s,i=new vS(r.shape,o.shape,a);return t.runWebGLProgram(i,[r],r.dtype)}const IS={kernelName:Rh,backendName:"webgl",kernelFunc:SS};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class TS{constructor(e,t,s,o,r){this.variableNames=["A"],this.outputShape=[];const[a,i,c,l]=e;this.outputShape=[a,t,s,l];const u=[o&&t>1?i-1:i,o&&s>1?c-1:c],d=[o&&t>1?t-1:t,o&&s>1?s-1:s],h=o?"0.5":"0.0";let f;r?f="max((vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC, vec2(0.0))":f="vec2(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${u[0]/d[0]},
          ${u[1]/d[1]});
      const vec2 inputShapeRC = vec2(${i}.0, ${c}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${f};

        // Compute the coordinators of nearest neighbor point.
        ivec2 sourceNearestRC = ivec2(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${h})));
        float newValue = getA(b, sourceNearestRC.x, sourceNearestRC.y, d);

        setOutput(newValue);
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class RS{constructor(e,t,s,o,r){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];const[a,i,c,l]=e;this.outputShape=[a,t,s,l];const u=[o&&t>1?i-1:i,o&&s>1?c-1:c],d=[o&&t>1?t-1:t,o&&s>1?s-1:s],h=o?"0.5":"0.0";let f;r?f="max((vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC, vec3(0.0))":f="vec3(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${u[0]/d[0]},
          ${u[1]/d[1]},
          ${u[1]/d[1]});
      const vec3 inputShapeRC = vec3(${i}.0, ${c}.0,
                                     ${c}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${f};

        // Compute the coordinators of nearest neighbor point.
        ivec3 sourceNearestRC = ivec3(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${h})));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${l-1};
        bool hasNextRow = coords.z < ${s-1};

        vec4 newValue = vec4(
          getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d),
          hasNextCol ? getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d + 1) : 0.0);

        setOutput(newValue);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ES(n){const{inputs:e,backend:t,attrs:s}=n,{images:o}=e,{alignCorners:r,halfPixelCenters:a,size:i}=s,[c,l]=i,u=$().getBool("WEBGL_PACK_IMAGE_OPERATIONS")?new RS(o.shape,c,l,r,a):new TS(o.shape,c,l,r,a);return t.runWebGLProgram(u,[o],o.dtype)}const kS={kernelName:Sh,backendName:"webgl",kernelFunc:ES};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class NS{constructor(e,t,s){this.variableNames=["dy"],this.outputShape=[],this.outputShape=t;const[,o,r]=t,[,a,i]=e,c=[s&&a>1?o-1:o,s&&i>1?r-1:r],l=[s&&a>1?a-1:a,s&&i>1?i-1:i],u=c[0]/l[0],d=c[1]/l[1],h=1/u,f=1/d,p=Math.ceil(h)*2+2,x=Math.ceil(f)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${u});
        const float widthScale = float(${d});

        const float invHeightScale = float(${h});
        const float invWidthScale = float(${f});

        const int winHeight = int(${p});
        const int winWidth = int(${x});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(floor(startRLerp - float(winHeight / 2)));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(floor(startCLerp - float(winWidth / 2)));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${a}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${i}) {
              continue;
            }

            float sourceFracRow =
              float(${c[0]}) *
                (float(dyR) / float(${l[0]}));

            float sourceFracCol =
                float(${c[1]}) *
                  (float(dyC) / float(${l[1]}));

            int sourceNearestRow = int(min(
                float(int(${o}) - 1),
                ${s} ? float(round(sourceFracRow)) :
                                  float(floor(sourceFracRow))));

            int sourceNearestCol = int(min(
                float(int(${r}) - 1),
                ${s} ? float(round(sourceFracCol)) :
                                  float(floor(sourceFracCol))));

            if (r == sourceNearestRow && c == sourceNearestCol) {
              accumulator += getDy(b, dyR, dyC, d);
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function AS(n){const{inputs:e,backend:t,attrs:s}=n,{images:o,dy:r}=e,{alignCorners:a}=s,i=new NS(r.shape,o.shape,a);return t.runWebGLProgram(i,[r],r.dtype)}const FS={kernelName:Ih,backendName:"webgl",kernelFunc:AS};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class DS{constructor(e,t){this.variableNames=["x"];const s=e.length;if(s>4)throw new Error(`WebGL backend: Reverse of rank-${s} tensor is not yet supported`);if(this.outputShape=e,s===1){this.userCode=`
        void main() {
          int coord = getOutputCoords();
          setOutput(getX(${e[0]} - coord - 1));
        }
      `;return}const o=i=>t.indexOf(i)!==-1&&e[i]!==1?`${e[i]} - coords[${i}] - 1`:`coords[${i}]`,r=e.map((i,c)=>o(c)).join(","),a=H(s);this.userCode=`
      void main() {
        ${a} coords = getOutputCoords();
        setOutput(getX(${r}));
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class OS{constructor(e,t){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0;const s=e.length;if(s>4)throw new Error(`WebGL backend: Reverse of rank-${s} tensor is not yet supported`);this.outputShape=e;const o=pe("rc",s),r=`${o[s-1]} + 1 < ${this.outputShape[s-1]}`,a=`${o[s-2]} + 1 < ${this.outputShape[s-2]}`,i=H(s);s===1?this.userCode=`
        void main(){
          int rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = getChannel(getX(${e[0]} - rc - 1),
            ${e[0]} - rc - 1);
          if(${r}){
              result.g = getChannel(getX(${e[0]} - (rc  + 1) - 1),
                ${e[0]} - (rc  + 1) - 1);
          }
          setOutput(result);
        }
      `:this.userCode=`
        void main() {
          ${i} rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = ${c(o.slice())};
          if(${r}){
            result.g = ${l(o.slice())};
          }
          if(${a}) {
            result.b = ${u(o.slice())};
            if(${r}) {
              result.a = ${d(o.slice())};
            }
          }
          setOutput(result);
        }
    `;function c(p){return h(p)}function l(p){return p[s-1]="("+p[s-1]+" + 1)",h(p)}function u(p){return p[s-2]="("+p[s-2]+" + 1)",h(p)}function d(p){return p[s-1]="("+p[s-1]+" + 1)",p[s-2]="("+p[s-2]+" + 1)",h(p)}function h(p){const x=e.map((b,C)=>f(C,p)),g=x.join(","),m=x.slice(-2).join(",");return`getChannel(getX(${g}), vec2(${m}))`}function f(p,x){return t.indexOf(p)!==-1&&e[p]!==1?`${e[p]} - ${x[p]} - 1`:`${x[p]}`}}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function PS(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{dims:r}=s,a=o.shape.length,i=ce(r,o.shape);if(a===0)return ye({inputs:{x:o},backend:t});const c=$().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new OS(o.shape,i):new DS(o.shape,i);return t.runWebGLProgram(c,[o],o.dtype)}const _S={kernelName:Eh,backendName:"webgl",kernelFunc:PS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class LS{constructor(e,t){this.variableNames=["Image"],this.outputShape=[],this.customUniforms=[{name:"params",type:"vec4"}];const s=e[1],o=e[2];this.outputShape=e;let r="";typeof t=="number"?r=`float outputValue = ${t.toFixed(2)};`:r=`
        vec3 fill = vec3(${t.join(",")});
        float outputValue = fill[coords[3]];`,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];
          int y = coords[1];
          float coordXFloat = (float(x) - params[0]) * params[3] -
            (float(y) - params[1]) * params[2];
          float coordYFloat = (float(x) - params[0]) * params[2] +
            (float(y) - params[1]) * params[3];
          int coordX = int(round(coordXFloat + params[0]));
          int coordY = int(round(coordYFloat + params[1]));
          ${r}
          if(coordX >= 0 && coordX < ${o} && coordY >= 0 && coordY < ${s}) {
            outputValue = getImage(coords[0], coordY, coordX, coords[3]);
          }
          setOutput(outputValue);
        }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const BS={kernelName:af,backendName:"webgl",kernelFunc:({inputs:n,attrs:e,backend:t})=>{const{image:s}=n,{radians:o,fillValue:r,center:a}=e,i=t,c=new LS(s.shape,r),[l,u]=Xi(a,s.shape[1],s.shape[2]),d=[[l,u,Math.sin(o),Math.cos(o)]];return i.runWebGLProgram(c,[s],s.dtype,d)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const MS=`
  // OpenGL ES does not support round function.
  // The algorithm is based on banker's rounding.
  float base = floor(x);
  if ((x - base) < 0.5) {
    return floor(x);
  } else if ((x - base) > 0.5) {
    return ceil(x);
  } else {
    if (mod(base, 2.0) == 0.0) {
      return base;
    } else {
      return base + 1.0;
    }
  }
`,VS=U({opSnippet:MS}),US={kernelName:kh,backendName:"webgl",kernelFunc:VS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const WS="return inversesqrt(x);",GS=U({opSnippet:WS,cpuKernelImpl:eb}),zS={kernelName:Ao,backendName:"webgl",kernelFunc:GS};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class wr{constructor(e,t,s,o,r,a,i=!0,c=!1){this.variableNames=["updates","indices","defaultValue"],this.outputShape=a;const l=H(r.length),u=H(a.length);let d="";s===1?d="i":s===2&&(d="i, j");const h=`getIndices(${d})`;let f="";o===1?f="i":o===2&&(f="i, coords[1]");const p=`getUpdates(${f})`;let x="";c&&(x="coords[0], coords[1]");const g=`getDefaultValue(${x})`,m=t>1?"strides[j]":"strides";this.userCode=`
        ${l} strides = ${l}(${r});

        void main() {
          ${u} coords = getOutputCoords();
          float sum = 0.0;
          bool found = false;
          for (int i = 0; i < ${e}; i++) {
            int flattenedIndex = 0;
            for (int j = 0; j < ${t}; j++) {
              int index = round(${h});
              flattenedIndex += index * ${m};
            }
            if (flattenedIndex == coords[0]) {
              sum += ${p};
              found = true;
            }
          }
          setOutput(mix(${g}, sum, float(found)));
        }
      `}}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class HS{constructor(e,t,s,o,r,a,i=!0,c=!1){this.variableNames=["updates","indices","defaultValue"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=a;const l=H(r.length),u=H(a.length);let d="";s===1?d="i":s===2&&(d="i, j");const h=`getIndices(${d})`;let f="";o===1?f="i":o===2&&(f="i, coords[1]");const p=`getUpdates(${f})`;let x="";c&&(x="coords[0], coords[1]");const g=`getDefaultValue(${x})`,m=t>1?"strides[j]":"strides",b=t>1?"strides[j + 1]":"strides";this.userCode=`
        ${l} strides = ${l}(${r});

        void main() {
          ${u} coords = getOutputCoords();
          vec4 sum = vec4(0.);
          vec4 found = vec4(0.);
          for (int i = 0; i < ${e}; i+=2) {
            ivec2 flattenedIndex = ivec2(0);
            for (int j = 0; j < ${t}; j+=2) {
              ivec4 index = round(${h});
              flattenedIndex += index.xz * ${m};
              if (j + 1 < ${t}) {
                flattenedIndex += index.yw * ${b};
              }
            }
            if (flattenedIndex[0] == coords[0] || flattenedIndex[1] == coords[0] ||
                flattenedIndex[0] == coords[0] + 1 || flattenedIndex[1] == coords[0] + 1) {
              vec4 updVals = ${p};
              if (flattenedIndex[0] == coords[0]) {
                sum.xy += updVals.xy;
                found.xy = vec2(1.);
              } else if (flattenedIndex[0] == coords[0] + 1) {
                sum.zw += updVals.xy;
                found.zw = vec2(1.);
              }
              if (flattenedIndex[1] == coords[0]) {
                sum.xy += updVals.zw;
                found.xy = vec2(1.);
              } else if (flattenedIndex[1] == coords[0] + 1) {
                sum.zw += updVals.zw;
                found.zw = vec2(1.);
              }
            }
          }
          setOutput(mix(${g}, sum, found));
        }
      `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function XS(n){const{inputs:e,backend:t,attrs:s}=n,{indices:o,updates:r}=e,{shape:a}=s,{sliceRank:i,numUpdates:c,sliceSize:l,strides:u,outputSize:d}=Mn(r,o,a),h=[d/l,l];if(d===0)return t.makeTensorInfo(a,o.dtype);const f=F({inputs:{x:o},backend:t,attrs:{shape:[c,i]}}),p=F({inputs:{x:r},backend:t,attrs:{shape:[c,l]}}),x=t.makeTensorInfo([],"float32",new Float32Array([0]));let g;$().getBool("WEBGL_PACK")?g=new HS(c,i,f.shape.length,p.shape.length,u,h):g=new wr(c,i,f.shape.length,p.shape.length,u,h);const m=t.runWebGLProgram(g,[p,f,x],p.dtype),b=F({inputs:{x:m},backend:t,attrs:{shape:a}});return t.disposeIntermediateTensorInfo(f),t.disposeIntermediateTensorInfo(p),t.disposeIntermediateTensorInfo(m),t.disposeIntermediateTensorInfo(x),b}const qS={kernelName:Nh,backendName:"webgl",kernelFunc:XS};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class jS{constructor(e,t,s,o){this.variableNames=["sortedSequence","values"],this.customUniforms=[{name:"numInputs",type:"int"}],this.outputShape=[e,s];const r="while (left < right) {",a=`for (int i = 0; i < ${Math.ceil(Math.log2(t+1))}; ++i) { if (left >= right) break;`,i=$().getNumber("WEBGL_VERSION")===2?r:a,c=o==="left"?"<":"<=";this.userCode=`
       int findBound(int batch, float value) {
         int left = 0;
         int right = numInputs;
         int mid;
         ${i}
           mid = (left + right) / 2;
           if (getSortedSequence(batch, mid) ${c} value) {
             left = mid + 1;
           } else {
             right = mid;
           }
         }
         return right;
       }

       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int valueIndex = coords[1];

         float value = getValues(batch, valueIndex);

         setOutput(float(findBound(batch, value)));
       }
     `}}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function KS(n){const{inputs:e,backend:t,attrs:s}=n,{sortedSequence:o,values:r}=e,{side:a}=s,i=new jS(o.shape[0],o.shape[1],r.shape[1],a),c=[[o.shape[1]]];return t.runWebGLProgram(i,[o,r],"int32",c)}const YS={kernelName:Fh,backendName:"webgl",kernelFunc:KS};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class QS{constructor(e,t,s){this.variableNames=["c","a","b"],this.outputShape=t;let o,r;if(s>4)throw Error(`Where for rank ${s} is not yet supported`);if(s===1)r="resRC",o="resRC";else{const i=["resRC.x","resRC.y","resRC.z","resRC.w"],c=[],l=[];for(let u=0;u<t.length;u++)l.push(`${i[u]}`),u<e&&c.push(`${i[u]}`);o=c.join(),r=l.join()}const a=H(s);this.userCode=`
      void main() {
        ${a} resRC = getOutputCoords();
        float cVal = getC(${o});
        if (cVal >= 1.0) {
          setOutput(getA(${r}));
        } else {
          setOutput(getB(${r}));
        }
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ZS(n){const{inputs:e,backend:t}=n,{condition:s,t:o,e:r}=e,a=new QS(s.shape.length,o.shape,o.shape.length);return t.runWebGLProgram(a,[s,o,r],Ge(o.dtype,r.dtype))}const JS={kernelName:Dh,backendName:"webgl",kernelFunc:ZS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const eI=`
  // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
  // see: https://arxiv.org/abs/1706.02515
  float scaleAlpha = ${Ki};
  float scale = ${Yi};
  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);
`,tI=U({opSnippet:eI}),nI={kernelName:Oh,backendName:"webgl",kernelFunc:tI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sI=ln+`
  return 1.0 / (1.0 + exp(-1.0 * x));
`,oI=`
  vec4 result = 1.0 / (1.0 + exp(-1.0 * x));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,rI=U({opSnippet:sI,packedOpSnippet:oI,cpuKernelImpl:nb}),aI={kernelName:ys,backendName:"webgl",kernelFunc:rI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const iI=`
  if (isnan(x)) { return 0.0; }
  return sign(x);
`,cI=U({opSnippet:iI}),lI={kernelName:Lh,backendName:"webgl",kernelFunc:cI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uI=ln+`
  return sin(x);
`,dI=`
  vec4 result = sin(x);
  bvec4 isNaN = isnan(x);
  ${Pt}
  return result;
`,hI=U({opSnippet:uI,packedOpSnippet:dI}),fI={kernelName:Ph,backendName:"webgl",kernelFunc:hI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pI=`
  float e2x = exp(x);
  return (e2x - 1.0 / e2x) / 2.0;
`,mI=U({opSnippet:pI}),gI={kernelName:_h,backendName:"webgl",kernelFunc:mI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xI=`
  float epsilon = 1.1920928955078125e-7;
  float threshold = log(epsilon) + 2.0;

  bool too_large = x > -threshold;
  bool too_small = x < threshold;

  float result;
  float exp_x = exp(x);

  if (too_large){
    result = x;
  }
  else if (too_small){
    result = exp_x;
  }
  else{
    result = log(exp_x + 1.0);
  }
  return result;
`,bI=U({opSnippet:xI}),wI={kernelName:Bh,backendName:"webgl",kernelFunc:bI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const CI=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{blockShape:r,paddings:a}=s;R(o.shape.length<=4,()=>"spaceToBatchND for rank > 4 with a WebGL backend not implemented yet");const i=r.reduce((m,b)=>m*b),c=[[0,0]];c.push(...a);for(let m=1+r.length;m<o.shape.length;++m)c.push([0,0]);const l=[],u=Su({inputs:{x:o},backend:t,attrs:{paddings:c,constantValue:0}}),d=er(u.shape,r,i,!1),h=tr(d.length,r.length,!1),f=nr(u.shape,r,i,!1),p=F({inputs:{x:u},backend:t,attrs:{shape:d}}),x=me({inputs:{x:p},backend:t,attrs:{perm:h}}),g=F({inputs:{x},backend:t,attrs:{shape:f}});return l.push(u),l.push(p),l.push(x),l.forEach(m=>t.disposeIntermediateTensorInfo(m)),g},yI={kernelName:Mh,backendName:"webgl",kernelFunc:CI};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $I(n){const{inputs:e,backend:t}=n,{indices:s,values:o,denseShape:r,defaultValue:a}=e;if(r.shape.length!==1)throw new Error(`Dense shape must be a vector, saw:
         ${r.shape}`);if(s.shape.length!==2)throw new Error(`Indices must be a matrix, saw:
         ${s.shape}`);if(o.shape.length!==1)throw new Error(`Values must be a vector, saw:
         ${o.shape}`);if(a.shape.length!==0)throw new Error(`Default value must be a scalar, saw:
        ${a.shape}`);const i=t.readSync(s.dataId),c=t.readSync(o.dataId),l=t.readSync(r.dataId),u=t.readSync(a.dataId)[0],[d,h,f,p,x]=ob(i,s.shape,s.dtype,c,o.dtype,l,u);return[t.makeTensorInfo(h,s.dtype,d),t.makeTensorInfo([h[0]],o.dtype,f),t.makeTensorInfo([p.length],"bool",new Uint8Array(p.map(g=>Number(g)))),t.makeTensorInfo([x.length],s.dtype,new Int32Array(x))]}const vI={kernelName:Wh,backendName:"webgl",kernelFunc:$I};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function SI(n){const{inputs:e,backend:t}=n,{inputIndices:s,inputShape:o,newShape:r}=e;if(s.shape.length!==2)throw new Error(`Input indices should be a matrix but received shape ${s.shape}`);if(o.shape.length!==1)throw new Error(`Input shape should be a vector but received shape ${o.shape}`);if(r.shape.length!==1)throw new Error(`Target shape should be a vector but received shape ${r.shape}`);const a=Array.from(t.readSync(o.dataId)),i=t.readSync(s.dataId),c=Array.from(t.readSync(r.dataId)),[l,u,d]=rb(i,s.shape,s.dtype,a,c);return[t.makeTensorInfo(u,s.dtype,l),t.makeTensorInfo([d.length],r.dtype,new Int32Array(d))]}const II={kernelName:Gh,backendName:"webgl",kernelFunc:SI};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function TI(n){const{inputs:e,backend:t}=n,{data:s,indices:o,segmentIds:r}=e;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.shape.length!==1)throw new Error(`Indices should be a vector but received shape
              ${o.shape}`);if(r.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
              ${r.shape}`);const a=t.readSync(s.dataId),i=t.readSync(o.dataId),c=t.readSync(r.dataId),[l,u]=jl(a,s.shape,s.dtype,i,c,!0);return t.makeTensorInfo(u,s.dtype,l)}const RI={kernelName:zh,backendName:"webgl",kernelFunc:TI};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function EI(n){const{inputs:e,backend:t}=n,{data:s,indices:o,segmentIds:r}=e;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.shape.length!==1)throw new Error(`Indices should be a vector but received shape
             ${o.shape}`);if(r.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
             ${r.shape}`);const a=t.readSync(s.dataId),i=t.readSync(o.dataId),c=t.readSync(r.dataId),[l,u]=jl(a,s.shape,s.dtype,i,c);return t.makeTensorInfo(u,s.dtype,l)}const kI={kernelName:Hh,backendName:"webgl",kernelFunc:EI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function NI(n){const{inputs:e,backend:t,attrs:s}=n,{sparseIndices:o,sparseValues:r,defaultValue:a}=e,{outputShape:i}=s,{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:h}=Mn(r,o,i),f=!1;if(r.dtype==="string"){const m=t.bufferSync(o),b=t.bufferSync(r),C=Ct(t.readSync(a.dataId)[0]),y=tb(m,b,i,h,u,l,c,d,C,f);return t.makeTensorInfo(i,y.dtype,y.values)}const p=new wr(l,c,o.shape.length,r.shape.length,d,[h,1],f),x=t.runWebGLProgram(p,[r,o,a],r.dtype),g=F({inputs:{x},backend:t,attrs:{shape:i}});return t.disposeIntermediateTensorInfo(x),g}const AI={kernelName:Xh,backendName:"webgl",kernelFunc:NI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function FI(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{numOrSizeSplits:r,axis:a}=s,i=ce(a,o.shape)[0],c=cc(o,r,i),l=o.shape.length,u=new Array(l).fill(0),d=o.shape.slice();return c.map(h=>{const f=[...d];f[i]=h;const p=un({inputs:{x:o},backend:t,attrs:{begin:u,size:f}});return u[i]+=h,p})}const DI={kernelName:Vh,backendName:"webgl",kernelFunc:FI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fa="return sqrt(x);",OI=U({opSnippet:fa,packedOpSnippet:fa,cpuKernelImpl:ab}),PI={kernelName:$s,backendName:"webgl",kernelFunc:OI};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _I="return x * x;",LI=U({opSnippet:_I}),BI={kernelName:qh,backendName:"webgl",kernelFunc:LI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pa="return (a - b) * (a - b);",MI=le({opSnippet:pa,packedOpSnippet:pa}),VI={kernelName:Fo,backendName:"webgl",kernelFunc:MI};/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function UI(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e;if(o.dtype!=="string")throw new Error("Input must be of datatype string");const r=t.readSync(o.dataId),a=Ye(r),i=ib(a,"string",s);return t.makeTensorInfo(o.shape,"string",i)}const WI={kernelName:Do,backendName:"webgl",kernelFunc:UI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function GI({inputs:n,attrs:e,backend:t}){const{x:s}=n,o=Fe+`
    return x > 0.0 ? 1.0 : float(${e.alpha});
  `,r=new Be(s.shape,o);return t.runWebGLProgram(r,[s],s.dtype)}const zI={kernelName:qa,backendName:"webgl",kernelFunc:GI};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class HI{constructor(e,t,s){this.variableNames=["x"],this.outputShape=s;const o=s.length,r=H(s.length),a=H(s.length);let i="";if(o===1)i="coords * strides + begin";else{let c=0;i=s.map((l,u)=>(c++,s.length===1?`coords * strides[${u}] + begin[${u}]`:`coords[${c-1}] * strides[${u}] + begin[${u}]`)).join(",")}this.userCode=`
      ${r} begin = ${r}(${e});
      ${r} strides = ${r}(${t});

      void main() {
        ${a} coords = getOutputCoords();
        setOutput(getX(${i}));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function XI(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{begin:r,end:a,strides:i,beginMask:c,endMask:l,ellipsisMask:u,newAxisMask:d,shrinkAxisMask:h}=s,{finalShapeSparse:f,finalShape:p,isIdentity:x,sliceDim0:g,isSimpleSlice:m,begin:b,end:C,strides:y}=Vi(o.shape,r,a,i,c,l,u,d,h);let S;if(x)S=F({inputs:{x:o},backend:t,attrs:{shape:p}});else if(g||m){R(o.shape.length>=1,()=>`Input must have rank at least 1, got: ${o.shape.length}`);const k=Ai(b,C,y),T=un({inputs:{x:o},backend:t,attrs:{begin:b,size:k}});S=F({inputs:{x:T},backend:t,attrs:{shape:p}}),t.disposeIntermediateTensorInfo(T)}else if(t.shouldExecuteOnCPU([o])){const T=t.readSync(o.dataId),I=ie(o.shape,o.dtype,T),v=cb(f,I,y,b);S=t.makeTensorInfo(p,o.dtype,v.values)}else{const T=new HI(b,y,f);S=t.runWebGLProgram(T,[o],o.dtype)}const N=F({inputs:{x:S},backend:t,attrs:{shape:p}});return t.disposeIntermediateTensorInfo(S),N}const qI={kernelName:jh,backendName:"webgl",kernelFunc:XI};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jI(n){const{inputs:e,backend:t,attrs:s}=n,{separator:o,nGramWidths:r,leftPad:a,rightPad:i,padWidth:c,preserveShortSequences:l}=s,{data:u,dataSplits:d}=e,h=t.readSync(u.dataId),f=t.readSync(d.dataId),[p,x]=lb(h,f,o,r,a,i,c,l);return[t.makeTensorInfo([p.length],"string",p),t.makeTensorInfo(d.shape,"int32",x)]}const KI={kernelName:Kh,backendName:"webgl",kernelFunc:jI};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function YI(n){const{inputs:e,backend:t,attrs:s}=n,{skipEmpty:o}=s,{input:r,delimiter:a}=e;if(r.dtype!=="string")throw new Error("Input must be of datatype string");if(r.shape.length!==1)throw new Error(`Input must be a vector, got shape: ${r.shape}`);if(a.shape.length!==0)throw new Error(`Delimiter must be a scalar, got shape: ${a.shape}`);const i=t.readSync(r.dataId),c=t.readSync(a.dataId)[0],[l,u,d]=ub(i,c,o),h=u.length;return[t.makeTensorInfo([h,2],"int32",l),t.makeTensorInfo([h],"string",u),t.makeTensorInfo([2],"int32",new Int32Array(d))]}const QI={kernelName:Yh,backendName:"webgl",kernelFunc:YI};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ZI(n){const{inputs:e,backend:t,attrs:s}=n,{numBuckets:o}=s,{input:r}=e;if(r.dtype!=="string")throw new Error("Input must be of datatype string");if(o<=0)throw new Error("Number of buckets must be at least 1");const a=t.readSync(r.dataId),i=db(a,o);return t.makeTensorInfo(r.shape,"int32",i)}const JI={kernelName:Qh,backendName:"webgl",kernelFunc:ZI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const eT="return tan(x);",tT=U({opSnippet:eT}),nT={kernelName:Zh,backendName:"webgl",kernelFunc:tT};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sT=`
  float e2x = exp(-2.0 * abs(x));
  return sign(x) * (1.0 - e2x) / (1.0 + e2x);
`,oT=U({opSnippet:sT}),rT={kernelName:Jh,backendName:"webgl",kernelFunc:oT};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function aT(n){const{inputs:e,backend:t,attrs:s}=n,{tensor:o,indices:r,updates:a}=e,{sliceRank:i,numUpdates:c,sliceSize:l,strides:u,outputSize:d}=Mn(a,r,o.shape),h=[d/l,l];if(d===0)return t.makeTensorInfo(o.shape,r.dtype);const f=F({inputs:{x:r},backend:t,attrs:{shape:[c,i]}}),p=F({inputs:{x:a},backend:t,attrs:{shape:[c,l]}}),x=F({inputs:{x:o},backend:t,attrs:{shape:h}}),g=new wr(c,i,f.shape.length,p.shape.length,u,h,!1,!0),m=t.runWebGLProgram(g,[p,f,x],x.dtype),b=F({inputs:{x:m},backend:t,attrs:{shape:o.shape}});return t.disposeIntermediateTensorInfo(f),t.disposeIntermediateTensorInfo(p),t.disposeIntermediateTensorInfo(x),t.disposeIntermediateTensorInfo(m),b}const iT={kernelName:Ah,backendName:"webgl",kernelFunc:aT};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class cT{constructor(e,t){this.variableNames=["A"];const s=new Array(e.length);for(let a=0;a<s.length;a++)s[a]=e[a]*t[a];this.outputShape=s,this.rank=s.length;const o=H(this.rank),r=lT(e);this.userCode=`
      void main() {
        ${o} resRC = getOutputCoords();
        setOutput(getA(${r}));
      }
    `}}function lT(n){const e=n.length;if(e>5)throw Error(`Tile for rank ${e} is not yet supported`);if(e===1)return`imod(resRC, ${n[0]})`;const t=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u"],s=[];for(let o=0;o<n.length;o++)s.push(`imod(${t[o]}, ${n[o]})`);return s.join()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tu(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{reps:r}=s;if(o.dtype==="string"||o.shape.length>5){const c=t.readSync(o.dataId),l=o.dtype==="string"?c.map(h=>Ct(h)):c,u=ie(o.shape,o.dtype,l),d=fb(u,r);return t.makeTensorInfo(d.shape,d.dtype,d.values)}const a=new cT(o.shape,r);return t.runWebGLProgram(a,[o],o.dtype)}const uT={kernelName:za,backendName:"webgl",kernelFunc:Tu};class dT{constructor(e){this.variableNames=["x","indices"],this.customUniforms=[{name:"n",type:"int"},{name:"firstPass",type:"int"},{name:"negativeInf",type:"float"},{name:"dir",type:"int"},{name:"inc",type:"int"}],this.outputShape=e,this.userCode=`
       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // We compare elements pair-wise within a group of size 2 * inc.
         // The comparing rule for each group alternates between ascending
         // and descending. Within each group, we compare each pair at
         // positions i and i+inc. To decide whether an element at position i
         // is x0 or x1, we mod it by 2 * inc, if the result is smaller than
         // inc, it is in the first half of the group, we denote it as x0,
         // otherwise we denote it as x1.
         // For example, as shown in the Bitonic top K paper referenced above,
         // Figure5(a) shows that element[1] is in the
         // second half of the group when group size is 2, but it is in the
         // first half of the group when group size is 4.

         bool isFirstInPair = imod(elemIdx, 2 * inc) < inc;
         int i = isFirstInPair ? elemIdx : elemIdx - inc;

         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + inc : int(getIndices(batch, i + inc));
         float x0 = i0 < n ? getX(batch, i0) : negativeInf;
         float x1 = i1 < n ? getX(batch, i1) : negativeInf;

         // Denotes which direction indices are in (ascending or descending).
         bool reverse = imod(elemIdx, 2 * dir) >= dir;
         bool isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
         if (reverse == isGreater) { // Elements in opposite order of direction
           int iTemp = i0;
           i0 = i1;
           i1 = iTemp;
         }
         if (isFirstInPair) {
            setOutput(float(i0));
         } else {
            setOutput(float(i1));
         }
       }
     `}}class hT{constructor(e){this.variableNames=["x","indices"],this.customUniforms=[{name:"n",type:"int"},{name:"firstPass",type:"int"},{name:"k",type:"int"}],this.outputShape=e,this.userCode=`
    void main() {
         // Takes max of indices (0, k), (1, k + 1), (2, k + 2) ...
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // The output size is half of the previous size.
         // If the previous sequence is | | | | _ _ _ _  | | | |  _ _ _ _ (k=4),
         // we only need to output the indices at positions |, the indices at
         // positions _ can be thrown away, see Figure5(b) After Phase 2
         // (Merge phase) in the Bitonic Top K paper referenced above.
         // For example, the paper shows we only need to output the orange bars.
         // The output sequence should look like this | | | | | | | |.
         // Because the sequence is halved, to map the output index back
         // to the previous sequence to find the corresponding value,
         // we need to double the index. When we double the index,
         // we basically interpolate a position, so 2i looks like
         // | _ | _ | _ | _ | _ | _ | _. We move the | to the first k position
         // of each 2k positions by - elemIdx % k. E.g. for output at
         // index 4,5,6,7, we want to get the corresponding element at
         // original index 8,9,10,11, for output at index 8,9,10,11,
         // we want to get the corresponding element at original index
         // 16,17,18,19, so on and so forth.

         int i = elemIdx < k ? elemIdx : (elemIdx * 2 - imod(elemIdx, k));
         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + k : int(getIndices(batch, i + k));

         float x0 = getX(batch, i0);
         float x1 = i1 < n ? getX(batch, i1) : x0;

         setOutput(x0 >= x1 ? float(i0) : float(i1));
       }
     `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ft(n,e){e!==null&&n.disposeIntermediateTensorInfo(e)}function ma(n){let e=1;for(;e<n;)e*=2;return e}function fT(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{k:r,sorted:a}=s,i=$().getNumber("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD"),c=$().getNumber("TOPK_K_CPU_HANDOFF_THRESHOLD"),l=o.shape,u=l[l.length-1];if(t.shouldExecuteOnCPU([o])||u<i||r>c){const v=t.readSync(o.dataId),[w,E]=pb(v,l,o.dtype,r,a);return[t.makeTensorInfo(w.shape,w.dtype,w.values),t.makeTensorInfo(E.shape,E.dtype,E.values)]}if(r===0)return l[l.length-1]=0,[t.makeTensorInfo(l,o.dtype,[]),t.makeTensorInfo(l,"int32",[])];if(u===1)return[o,zn({attrs:{shape:l,dtype:"int32",value:0},backend:t})];const d=t.texData.get(o.dataId),h=d!==null&&d.isPacked,f=h?t.unpackTensor(o):o,x=A(l)/u,g=F({inputs:{x:f},attrs:{shape:[x,u]},backend:t});h&&ft(t,f);const m=ma(r),b=ma(u);let C=null;const y=()=>C===null?[g,g]:[g,C],S=(v,w,E)=>{const _=y(),B=new dT(E),W=[[u],[C===null?1:0],[Number.NEGATIVE_INFINITY],[v],[w]],j=C;C=t.runWebGLProgram(B,_,"int32",W),ft(t,j)};for(let v=1;v<m;v*=2){const w=v*2;for(let E=v;E>=1;E/=2)S(w,E,[x,b])}for(let v=b;v>m;v/=2){const w=y(),E=new hT([x,v/2]),B=[[u],[C===null?1:0],[m]],L=C;C=t.runWebGLProgram(E,w,"int32",B),ft(t,L);const W=m/2,j=W*2;for(let G=W;G>=1;G/=2)S(j,G,C.shape)}let N=C;C=un({inputs:{x:C},backend:t,attrs:{begin:0,size:[x,r]}}),ft(t,N);let k=bu({inputs:{x:g,indices:C},backend:t,attrs:{axis:1,batchDims:1}});ft(t,g);const T=l.slice(0,-1);T.push(r),N=C,C=F({inputs:{x:C},attrs:{shape:T},backend:t}),ft(t,N);const I=k;return k=F({inputs:{x:k},attrs:{shape:T},backend:t}),ft(t,I),[k,C]}const pT={kernelName:ef,backendName:"webgl",kernelFunc:fT};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class mT{constructor(e,t,s,o,r,a){this.variableNames=["Image","Transforms"],this.outputShape=a;const i=s==="nearest"?1:2;let c;switch(o){case"constant":c=1;break;case"reflect":c=2;break;case"wrap":c=3;break;case"nearest":c=4;break;default:c=1;break}this.userCode=`
            float mapCoord(float outCoord, float len) {
              float inCoord = outCoord;
              if(${c} == 2) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    if (inCoord < sz2) {
                      inCoord = sz2 * float(int(float(-inCoord / sz2))) +
                      inCoord;
                    }
                    inCoord = inCoord < -len ? inCoord + sz2 : -inCoord - 1.0;
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    inCoord -= sz2 * float(int(float(inCoord / sz2)));
                    if (inCoord >= len) {
                      inCoord = sz2 - inCoord - 1.0;
                    }
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${c} == 3) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord += len * (float(int(float(-inCoord / sz))) + 1.0);
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord -= len * float(int(float(inCoord / sz)));
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${c} == 4) {
                return clamp(outCoord, 0.0, len - 1.0);
              } else {
                return outCoord;
              }
            }

            float readWithFillValue(int batch, int coordY, int coordX,
              int channel) {
              float outputValue;
              if (0 <= coordY && coordY < ${e} && 0 <= coordX && coordX < ${t}) {
                  outputValue = getImage(batch, coordY, coordX, channel);
              } else {
                outputValue = float(${r});
              }
              return outputValue;
            }

            void main() {
              ivec4 coords = getOutputCoords();
              float outputValue;
              int batch = coords[0];
              int x = coords[2];
              int y = coords[1];
              int channel = coords[3];
              float xf = float(x);
              float yf = float(y);
              float a1 = getTransforms(batch, 0);
              float a2 = getTransforms(batch, 1);
              float a3 = getTransforms(batch, 2);
              float b1 = getTransforms(batch, 3);
              float b2 = getTransforms(batch, 4);
              float b3 = getTransforms(batch, 5);
              float c1 = getTransforms(batch, 6);
              float c2 = getTransforms(batch, 7);
              float projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = float(${r});
              } else {
                float inX = (a1 * xf + a2 * yf + a3) / projection;
                float inY = (b1 * xf + b2 * yf + b3) / projection;
                float mapX = mapCoord(inX, float(${t}));
                float mapY = mapCoord(inY, float(${e}));

                if (${i} == 1) {
                  int coordY = int(round(mapY));
                  int coordX = int(round(mapX));
                  outputValue = readWithFillValue(batch, coordY, coordX,
                    channel);
                } else {
                  float yFloor = floor(mapY);
                  float xFloor = floor(mapX);
                  float yCeil = yFloor + 1.0;
                  float xCeil = xFloor + 1.0;
                  float valueYFloor = (xCeil - mapX) *
                  readWithFillValue(batch, int(yFloor), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yFloor), int(xCeil), channel);
                  float valueYCeil = (xCeil - mapX) *
                  readWithFillValue(batch, int(yCeil), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yCeil), int(xCeil), channel);
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutput(outputValue);
            }
        `}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gT(n){const{inputs:e,backend:t,attrs:s}=n,{image:o,transforms:r}=e,{interpolation:a,fillMode:i,fillValue:c,outputShape:l}=s,[u,d,h,f]=o.shape,[p,x]=l??[d,h],g=[u,p,x,f],m=new mT(d,h,a,i,c,g);return t.runWebGLProgram(m,[o,r],"float32")}const xT={kernelName:tf,backendName:"webgl",kernelFunc:gT};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bT(n){const{inputs:e,attrs:t,backend:s}=n,{axis:o}=t,{x:r}=e;nn(r,"unique"),console.warn("WARNING: ","UI might be locked temporarily as data is being downloaded");const a=s.readSync(r.dataId),{outputValues:i,outputShape:c,indices:l}=mb(a,o,r.shape,r.dtype);return[s.makeTensorInfo(c,r.dtype,i),s.makeTensorInfo([l.length],"int32",l)]}const wT={kernelName:nf,backendName:"webgl",kernelFunc:bT};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function CT(n){const{inputs:e,backend:t,attrs:s}=n,{value:o}=e;let{axis:r}=s;r<0&&(r+=o.shape.length);const a=o,i=a.shape.length,c=o.shape[r],l=new Array(i-1);let u=0;for(let x=0;x<i;x++)x!==r&&(l[u++]=a.shape[x]);const d=[],h=new Array(i).fill(0),f=a.shape.slice();f[r]=1;const p=new Array(c);for(let x=0;x<p.length;x++){h[r]=x;const g=un({inputs:{x:a},backend:t,attrs:{begin:h,size:f}}),m=F({inputs:{x:g},backend:t,attrs:{shape:l}});p[x]=m,d.push(g)}return d.forEach(x=>t.disposeIntermediateTensorInfo(x)),p}const yT={kernelName:sf,backendName:"webgl",kernelFunc:CT};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $T{constructor(e,t){this.variableNames=["x","segmentIds"];const s=e.windowSize,o=e.batchSize,r=e.inSize,a=e.numSegments,i=a*Math.ceil(r/s);this.outputShape=[o,i];const c="0.0",l="sumValue",u=Math.floor(s/4)*4,d=s%4,h=`
        sumValue += dot(values, segFilter);
    `;let f="";r%s>0&&(f=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return initializationValue;
        }
      `);let p="";r%s>0&&(p=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return -1.0;
        }
      `),this.userCode=`
      const float initializationValue = ${c};

      float getValue(int batch, int inIdx) {
        ${f}
        return getX(batch, inIdx);
      }

      float getSegmentIdAtIndex(int inIdx) {
        ${p}
        return getSegmentIds(inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = int(floor(float(outIdx) / float(
          ${a})) * float(${s}));
        int currentSeg = int(mod(float(outIdx), float(${a})));

        float sumValue = 0.0;

        for (int i = 0; i < ${u}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 3)) == currentSeg ? 1 : 0
          );

          ${h}
        }

        int inIdx = inOffset + ${u};
        if (${d===1}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          int inIdxSeg = int(getSegmentIdAtIndex(inIdx));

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            0,
            0,
            0
          );

          ${h}
        } else if (${d===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
              0,
              0
          );

          ${h}
        } else if (${d===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            0
          );

          ${h}
        }
        setOutput(${l});
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vT(n){const{inputs:e,backend:t,attrs:s}=n,{x:o,segmentIds:r}=e,{numSegments:a}=s,i=o.shape.length,c=[];let l=0;const u=Re([l],i);let d=o;u!=null&&(d=me({inputs:{x:o},backend:t,attrs:{perm:u}}),c.push(d),l=Ee(1,i)[0]);const h=yc(d.shape,l,a),f=A([d.shape[l]]),p=F({inputs:{x:d},backend:t,attrs:{shape:[-1,f]}});c.push(p);const x=Po(o.dtype),g=(y,S,N,k,T)=>{const I=y.shape[0],v=y.shape[1],w=Cc(v,T),E={windowSize:w,inSize:v,batchSize:I,numSegments:T},_=new $T(E,S),B=t.compileAndRun(_,[y,N],k);if(c.push(B),B.shape[1]===T)return B;const L=Iu({backend:t,attrs:{start:0,stop:T,step:1,dtype:"float32"}}),W=Tu({inputs:{x:L},backend:t,attrs:{reps:[v/w]}});return c.push(L),c.push(W),g(B,S,W,k,T)},m=g(p,"unsortedSegmentSum",r,x,a),b=F({inputs:{x:m},backend:t,attrs:{shape:h}});let C=b;if(u!=null){c.push(b);const y=zo(u);C=me({inputs:{x:C},backend:t,attrs:{perm:y}})}return c.forEach(y=>t.disposeIntermediateTensorInfo(y)),C}const ST={kernelName:of,backendName:"webgl",kernelFunc:vT};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const IT=[cw,uw,fw,gw,bw,yw,vw,Iw,kw,Aw,Ow,Lw,Vw,zw,qw,Kw,Qw,tC,sC,rC,lC,gC,bC,$C,SC,NC,FC,_C,Hb,MC,zC,jC,ey,sy,ry,iy,ly,fy,gy,wy,yy,vy,Iy,Ey,Ny,Oy,_y,My,Wy,zy,jy,Zy,n1,r1,c1,l1,d1,f1,m1,x1,w1,v1,T1,k1,A1,O1,L1,U1,H1,zb,q1,WC,Y1,J1,n$,qb,a$,u$,h$,g$,w$,v$,T$,N$,O$,L$,M$,G$,H$,q$,Q$,J$,tv,sv,rv,lv,fv,xv,Iv,Yb,kv,Fv,Pv,Bv,TC,Uv,Gv,Hv,jv,Zv,Kb,eS,nS,oS,aS,iS,RC,yv,uS,pS,bS,Zb,$S,IS,kS,FS,_S,BS,US,zS,qS,YS,JS,nI,aI,lI,fI,gI,pC,vv,wI,yI,vI,II,RI,kI,AI,DI,PI,BI,VI,WI,zI,qI,KI,QI,JI,$v,rw,nT,rT,iT,uT,pT,xT,aw,wT,yT,ST,Wv];for(const n of IT)ja(n);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rE=Object.freeze(Object.defineProperty({__proto__:null,GPGPUContext:ss,MathBackendWebGL:Wn,forceHalfFloat:Ql,gpgpu_util:px,setWebGLContext:Sc,version_webgl:Wb,webgl:Gb,webgl_util:Cg},Symbol.toStringTag,{value:"Module"}));export{Fd as $,ju as A,ud as B,md as C,Pp as D,O as E,dd as F,Vd as G,hd as H,xo as I,fd as J,bo as K,Bp as L,pd as M,Ap as N,gd as O,bd as P,wd as Q,yd as R,Wa as S,Jh as T,$d as U,vd as V,Sd as W,Id as X,Rd as Y,Ed as Z,kd as _,Ku as a,gh as a$,Dd as a0,ae as a1,Co as a2,Lp as a3,Dh as a4,Me as a5,Ve as a6,Od as a7,Lu as a8,_d as a9,Cn as aA,Jd as aB,Eo as aC,Kd as aD,Oa as aE,Bh as aF,CR as aG,zt as aH,Yd as aI,Qd as aJ,Zd as aK,A as aL,Fh as aM,nh as aN,oh as aO,ah as aP,ih as aQ,Qt as aR,eo as aS,li as aT,Ta as aU,xe as aV,ko as aW,lh as aX,uh as aY,dh as aZ,No as a_,th as aa,ch as ab,ce as ac,Xe as ad,Tp as ae,Xp as af,jt as ag,Or as ah,ct as ai,wt as aj,yo as ak,Ld as al,$o as am,za as an,ie as ao,vo as ap,Ud as aq,So as ar,Io as as,zd as at,Hd as au,Xd as av,qd as aw,To as ax,Ro as ay,jd as az,R as b,Td as b$,mh as b0,bh as b1,Mh as b2,Zt as b3,La as b4,wh as b5,Ch as b6,yh as b7,ot as b8,Fn as b9,Ah as bA,ef as bB,nf as bC,of as bD,sf as bE,im as bF,Kt as bG,ee as bH,Ha as bI,Nf as bJ,Nh as bK,Xh as bL,Wd as bM,Ue as bN,Ls as bO,xd as bP,dm as bQ,um as bR,_e as bS,lf as bT,cm as bU,Js as bV,lm as bW,Nd as bX,Ad as bY,uf as bZ,cf as b_,Te as ba,ii as bb,at as bc,$h as bd,Ba as be,vh as bf,Eh as bg,kh as bh,Ao as bi,Oh as bj,is as bk,Lh as bl,Ph as bm,_h as bn,Uh as bo,Bd as bp,Gd as bq,Vh as br,Fo as bs,lt as bt,xh as bu,jh as bv,Zh as bw,Pu as bx,ui as by,Si as bz,M as c,ka as c$,Md as c0,af as c1,hh as c2,gm as c3,ph as c4,bm as c5,fh as c6,xm as c7,Th as c8,Sh as c9,Br as cA,dR as cB,SR as cC,pR as cD,cR as cE,iR as cF,hR as cG,Bm as cH,NT as cI,$ as cJ,$r as cK,rf as cL,Ri as cM,Rm as cN,Em as cO,Im as cP,vm as cQ,Tm as cR,Sm as cS,go as cT,am as cU,On as cV,Wo as cW,ld as cX,id as cY,RT as cZ,gs as c_,tf as ca,Ie as cb,nm as cc,Wh as cd,Gh as ce,zh as cf,Hh as cg,Kh as ch,Yh as ci,Qh as cj,Do as ck,ze as cl,vR as cm,IR as cn,aR as co,fR as cp,Hf as cq,rR as cr,oR as cs,uR as ct,fi as cu,Qf as cv,Ss as cw,lR as cx,Dr as cy,Ni as cz,oe as d,Au as d$,Cd as d0,Re as d1,kT as d2,ET as d3,Aa as d4,Pd as d5,xs as d6,zo as d7,bs as d8,Da as d9,Xa as dA,LT as dB,ni as dC,Up as dD,$p as dE,jp as dF,em as dG,om as dH,Ne as dI,zf as dJ,Ii as dK,et as dL,Ti as dM,jm as dN,Sa as dO,Uf as dP,At as dQ,xa as dR,KT as dS,Ut as dT,Bf as dU,bp as dV,ms as dW,_o as dX,Af as dY,wf as dZ,ga as d_,FT as da,eh as db,ws as dc,rh as dd,sh as de,He as df,Cs as dg,Pa as dh,_a as di,Na as dj,Ua as dk,Ma as dl,Va as dm,Rh as dn,Ih as dp,Ki as dq,Yi as dr,ys as ds,Zo as dt,$s as du,qh as dv,qa as dw,vs as dx,Ga as dy,Zp as dz,Yu as e,tg as e$,Ze as e0,Ae as e1,In as e2,Vu as e3,Ct as e4,Ot as e5,Gf as e6,yl as e7,we as e8,lo as e9,$i as eA,te as eB,Ht as eC,_n as eD,Uu as eE,En as eF,co as eG,sc as eH,rc as eI,ac as eJ,oc as eK,ic as eL,Ux as eM,nc as eN,tc as eO,ec as eP,Ji as eQ,Zi as eR,Qi as eS,Zm as eT,Nt as eU,Km as eV,Ym as eW,Qm as eX,eg as eY,C0 as eZ,Jm as e_,o0 as ea,wa as eb,J as ec,bx as ed,Bl as ee,Ee as ef,Le as eg,We as eh,$e as ei,Ln as ej,er as ek,tr as el,nr as em,qi as en,ji as eo,r0 as ep,wx as eq,Ui as er,Tt as es,xl as et,Es as eu,vx as ev,en as ew,Bn as ex,Ge as ey,Cx as ez,Qu as f,oE as f$,Fa as f0,Yt as f1,jo as f2,kx as f3,$c as f4,Nx as f5,Px as f6,Ll as f7,Lx as f8,Dn as f9,AR as fA,OR as fB,ER as fC,PR as fD,_R as fE,LR as fF,BR as fG,MR as fH,VR as fI,UR as fJ,kR as fK,WR as fL,GR as fM,zR as fN,HR as fO,XR as fP,qR as fQ,jR as fR,KR as fS,QR as fT,NR as fU,ZR as fV,JR as fW,eE as fX,tE as fY,nE as fZ,sE as f_,Ix as fa,Qx as fb,Zx as fc,Jx as fd,e0 as fe,Xi as ff,Mn as fg,n0 as fh,a0 as fi,i0 as fj,c0 as fk,cc as fl,Vi as fm,Ai as fn,f0 as fo,m0 as fp,x0 as fq,b0 as fr,y0 as fs,$0 as ft,wn as fu,v0 as fv,Sx as fw,RR as fx,FR as fy,DR as fz,Zu as g,YR as g0,ja as g1,Wb as g2,wo as g3,Ra as g4,Wu as g5,ss as g6,AT as g7,DT as g8,Wn as g9,eR as gA,yR as gB,$R as gC,ZT as gD,sR as gE,Sc as gF,S0 as gG,Xm as gH,Po as gI,WT as gJ,QT as gK,MT as gL,BT as gM,UT as gN,bR as gO,wR as gP,Kp as gQ,Gb as gR,Cg as gS,rE as gT,OT as ga,PT as gb,Rr as gc,_T as gd,cs as ge,rg as gf,mR as gg,VT as gh,qT as gi,GT as gj,XT as gk,jT as gl,HT as gm,zT as gn,tR as go,nR as gp,Ql as gq,TR as gr,Wf as gs,vr as gt,Ws as gu,px as gv,gR as gw,xR as gx,YT as gy,JT as gz,Ju as h,ed as i,td as j,nd as k,sd as l,ut as m,rd as n,Q as o,od as p,Jt as q,Uo as r,Fp as s,ad as t,ls as u,cd as v,Lf as w,yi as x,z as y,V as z};
