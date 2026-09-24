// Keep the canonical Data Analytics portable builder and verifier, adding only
// a page-level containment rule for wide chart/table fallback content.
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [pluginRoot, inputPath, outputPath] = process.argv.slice(2);
const scriptRoot = resolve(pluginRoot, "skills/build-report/scripts");
const { buildPortableArtifact } = await import(
  pathToFileURL(resolve(scriptRoot, "build_portable_artifact.mjs"))
);
const { deliverPortableArtifact } = await import(
  pathToFileURL(resolve(scriptRoot, "deliver_portable_artifact.mjs"))
);
const receipt = await deliverPortableArtifact({ inputPath, outputPath }, {
  build: input => buildPortableArtifact(input).replace(
    "</head>",
    "<style>html,body{max-width:100%;overflow-x:hidden}</style>\n</head>",
  ),
});
process.stdout.write(JSON.stringify(receipt) + "\n");
if (!receipt.ok) process.exitCode = 1;
