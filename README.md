# UserToolBench project page

This folder is ready to publish with GitHub Pages.

## Files

- `index.html`: static project page
- `assets/profile_hidden_pipeline.svg`: profile-hidden pipeline diagram
- `assets/dataset_composition.svg`: dataset composition graphic
- `assets/performance_gap.svg`: exact-vs-relaxed performance graphic
- `assets/bottlenecks.svg`: bottleneck summary graphic
- `copy_snippets.md`: text snippets you can reuse in README, Hugging Face, or project docs

## How to publish

1. Create a GitHub repository, for example `UserToolBench`.
2. Upload these files to the repository root.
3. In GitHub, go to **Settings → Pages**.
4. Set **Source** to **Deploy from a branch**.
5. Select **Branch: main** and **Folder: /root**.
6. Replace the placeholder URLs in `index.html`:
   - `https://github.com/YOUR_USERNAME/UserToolBench`
   - `https://huggingface.co/datasets/YOUR_USERNAME/UserToolBench`
   - `paper.pdf`

## Optional

Put the paper PDF in the same folder as `index.html` and name it `paper.pdf`, or change the link in the HTML.
