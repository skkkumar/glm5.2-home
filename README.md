# glm5.2-home

The GLM-5.2 site: a guided film (home), the vLLM + AITER optimization map, and the original PR-review page.

Live URL once published: https://skkkumar.github.io/glm5.2-home/

## Pages

| Path | Page |
|---|---|
| `index.html` | Home: GLM-5.2, explained slowly (guided film with narration and animations) |
| `optimization/index.html` | GLM – optimization: where every vLLM and AITER PR sits in the model |
| `pr-review/index.html` | PR review: the original https://skkkumar.github.io/pr-review/ shown live inside this site |

`nav.css` and `nav.js` add the burger menu and side panel to every page.

## Publish (one time)

1. On GitHub, create a new **public** repository named exactly `glm5.2-home` (no README, no .gitignore).
2. In a terminal, inside this folder:

```bash
git init
git add .
git commit -m "GLM-5.2 site: film, optimization map, pr-review"
git branch -M main
git remote add origin https://github.com/skkkumar/glm5.2-home.git
git push -u origin main
```

3. In the repository: Settings → Pages → Build and deployment → Source: **Deploy from a branch** → Branch: **main**, folder **/ (root)** → Save.
4. Wait a minute, then open https://skkkumar.github.io/glm5.2-home/

The existing `pr-review` repository stays exactly as it is; this site only links to it and embeds it.

## Updating later

Edit the HTML files, then `git add . && git commit -m "update" && git push`. Pages redeploys automatically.

## Local preview

Open `index.html` in Chrome, Edge or Safari. All links use relative paths, so the menu works from a local folder too. (The PR-review sub page needs internet access, because it embeds the live page.)
