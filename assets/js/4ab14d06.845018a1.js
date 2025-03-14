"use strict";(self.webpackChunkmy_docu_website=self.webpackChunkmy_docu_website||[]).push([[7435],{9170:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>l,contentTitle:()=>o,default:()=>d,frontMatter:()=>s,metadata:()=>r,toc:()=>h});var i=n(4848),a=n(8453);const s={},o="Machine Learning Testing",r={id:"machine-learning/MLTesting",title:"Machine Learning Testing",description:"Sometimes people argue that machine learning is complicated because we can't know what it is doing under the hood. The truth is that many of our actual systems are so big that not a single person know what the system is doing all the time, we can't hold all this information in the head. So to bypass this problem we need to find a way to trust in our systems. This way is to have a comprehensive test suite for the software at hand. We do this already for many types of complex systems, why not do it for ML projects? Below I list the types of tests that I find useful working with.",source:"@site/docs/machine-learning/202103242105-MLTesting.md",sourceDirName:"machine-learning",slug:"/machine-learning/MLTesting",permalink:"/docs/machine-learning/MLTesting",draft:!1,unlisted:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/machine-learning/202103242105-MLTesting.md",tags:[],version:"current",sidebarPosition:202103242105,frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Improve Human Performance",permalink:"/docs/machine-learning/AiImprovePerformance"},next:{title:"Model Calibration",permalink:"/docs/machine-learning/ModelCalibration"}},l={},h=[];function c(e){const t={h1:"h1",header:"header",li:"li",p:"p",ul:"ul",...(0,a.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(t.header,{children:(0,i.jsx)(t.h1,{id:"machine-learning-testing",children:"Machine Learning Testing"})}),"\n",(0,i.jsx)(t.p,{children:"Sometimes people argue that machine learning is complicated because we can't know what it is doing under the hood. The truth is that many of our actual systems are so big that not a single person know what the system is doing all the time, we can't hold all this information in the head. So to bypass this problem we need to find a way to trust in our systems. This way is to have a comprehensive test suite for the software at hand. We do this already for many types of complex systems, why not do it for ML projects? Below I list the types of tests that I find useful working with."}),"\n",(0,i.jsx)(t.h1,{id:"unity-tests",children:"Unity tests"}),"\n",(0,i.jsx)(t.p,{children:"Mainly use unit tests to not let the model regress on a given and fixed problem. Example:"}),"\n",(0,i.jsxs)(t.ul,{children:["\n",(0,i.jsx)(t.li,{children:"We have a image sample that isn't working, we want to make sure that we will fix the problem and it will not happen again."}),"\n"]}),"\n",(0,i.jsx)(t.h1,{id:"integration-tests",children:"Integration tests"}),"\n",(0,i.jsx)(t.p,{children:"Mainly used to evaluate the model in the field, we must setup a comprehensive dataset that we can test all the possible difficulties that will happen to our model and then calculate something like precision and recall. So we can keep track of the evolution and data drift."})]})}function d(e={}){const{wrapper:t}={...(0,a.R)(),...e.components};return t?(0,i.jsx)(t,{...e,children:(0,i.jsx)(c,{...e})}):c(e)}},8453:(e,t,n)=>{n.d(t,{R:()=>o,x:()=>r});var i=n(6540);const a={},s=i.createContext(a);function o(e){const t=i.useContext(s);return i.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function r(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(a):e.components||a:o(e.components),i.createElement(s.Provider,{value:t},e.children)}}}]);