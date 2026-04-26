from docutils import nodes
from docutils.parsers.rst import Directive


MERMAID_JS = """
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.12.1/dist/mermaid.esm.min.mjs";

const currentTheme = () => {
  const explicitTheme = document.body.dataset.theme;
  if (explicitTheme === "dark") return "dark";
  if (explicitTheme === "light") return "default";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "default";
};

const restoreSources = () => {
  const elements = document.querySelectorAll(".mermaid");
  elements.forEach((element) => {
    element.dataset.mermaidSource ||= element.textContent;
    element.textContent = element.dataset.mermaidSource;
    element.removeAttribute("data-processed");
  });
  return elements.length > 0;
};

let renderPromise = Promise.resolve();
const render = () => {
  renderPromise = renderPromise.catch(() => {}).then(async () => {
    if (!restoreSources()) return;
    mermaid.initialize({ startOnLoad: false, theme: currentTheme() });
    await mermaid.run({ querySelector: ".mermaid" });
  });
  return renderPromise;
};

await render();

new MutationObserver(render).observe(document.body, {
  attributes: true,
  attributeFilter: ["data-theme"],
});
"""


class MermaidNode(nodes.General, nodes.Element):
    pass


class MermaidDirective(Directive):
    has_content = True

    def run(self):
        node = MermaidNode()
        node["code"] = "\n".join(self.content)
        return [node]


def visit_mermaid_html(self, node):
    self.body.append(f'<pre class="mermaid">{self.encode(node["code"])}</pre>')
    raise nodes.SkipNode


def visit_mermaid_text(self, node):
    self.add_text("[mermaid diagram]")
    raise nodes.SkipNode


def setup(app):
    app.add_node(
        MermaidNode,
        html=(visit_mermaid_html, None),
        text=(visit_mermaid_text, None),
    )
    app.add_directive("mermaid", MermaidDirective)
    app.add_js_file(None, body=MERMAID_JS, priority=300, type="module")

    return {"parallel_read_safe": True, "parallel_write_safe": True}
