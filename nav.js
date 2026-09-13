/* Shared side-panel navigation: burger button + slide-in panel.
   Usage: <script src="PATH/nav.js" data-root="PATH/" data-here="home|opt|pr"></script> */
(function(){
  var sc=document.currentScript, root=(sc&&sc.dataset.root)||'./', here=(sc&&sc.dataset.here)||'';
  var items=[
    ['home','Home','GLM-5.2, explained slowly — the guided film',root+'index.html'],
    ['opt','GLM – optimization','Where every vLLM and AITER PR sits in the model',root+'optimization/index.html'],
    ['pr','PR review','The original review-prep page (pr-review)',root+'pr-review/index.html']
  ];
  var btn=document.createElement('button'); btn.className='gnav-btn'; btn.setAttribute('aria-label','Open menu'); btn.setAttribute('aria-expanded','false');
  btn.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>';
  var ov=document.createElement('div'); ov.className='gnav-overlay';
  var panel=document.createElement('nav'); panel.className='gnav-panel'; panel.setAttribute('aria-label','Site menu');
  var html='<div class="gnav-head"><b>GLM-5.2 home</b><button class="gnav-close" aria-label="Close menu">×</button></div><ul class="gnav-list">';
  items.forEach(function(it){ html+='<li><a href="'+it[3]+'"'+(it[0]===here?' class="now" aria-current="page"':'')+'>'+it[1]+'<small>'+it[2]+'</small></a></li>'; });
  html+='</ul><div class="gnav-foot">Original host: <a href="https://skkkumar.github.io/pr-review/" target="_blank" rel="noopener">skkkumar.github.io/pr-review</a></div>';
  panel.innerHTML=html;
  document.body.appendChild(btn); document.body.appendChild(ov); document.body.appendChild(panel);
  function open(){ document.documentElement.classList.add('gnav-open'); btn.setAttribute('aria-expanded','true'); panel.querySelector('a').focus(); }
  function close(){ document.documentElement.classList.remove('gnav-open'); btn.setAttribute('aria-expanded','false'); btn.focus(); }
  btn.addEventListener('click',function(){ document.documentElement.classList.contains('gnav-open')?close():open(); });
  ov.addEventListener('click',close); panel.querySelector('.gnav-close').addEventListener('click',close);
  document.addEventListener('keydown',function(e){ if(e.key==='Escape'&&document.documentElement.classList.contains('gnav-open'))close(); });
})();
