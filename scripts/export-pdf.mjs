// Optional authoring dependency: npm install --no-save playwright
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const browser=await chromium.launch({headless:true,
  ...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
try {
  const page=await browser.newPage({viewport:{width:1280,height:720}});
  await page.goto(process.env.PRESENTATION_URL || 'http://127.0.0.1:8768/',{waitUntil:'networkidle'});
  await page.evaluate(async()=>{
    await document.fonts.ready;
    await Promise.all([...document.images].filter(i=>i.hasAttribute('src')).map(i=>i.decode()));
  });
  await page.emulateMedia({media:'print'});
  await page.pdf({path:path.join(root,'docs/assets/wand-of-death.pdf'),
    preferCSSPageSize:true,printBackground:true,displayHeaderFooter:false});
  console.log('Exported docs/assets/wand-of-death.pdf');
} finally {await browser.close();}
