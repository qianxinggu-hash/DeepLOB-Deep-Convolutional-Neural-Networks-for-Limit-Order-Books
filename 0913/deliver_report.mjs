// Use the canonical report delivery pipeline with the same scrollbar-width
// containment fix used by 0906. Charts, sources, styling and all verification
// remain owned by the Data Analytics portable builder.
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [pluginRoot, inputPath, outputPath] = process.argv.slice(2);
const scriptRoot = resolve(pluginRoot, "skills/build-report/scripts");
const { buildPortableArtifact } = await import(pathToFileURL(resolve(scriptRoot, "build_portable_artifact.mjs")));
const { deliverPortableArtifact } = await import(pathToFileURL(resolve(scriptRoot, "deliver_portable_artifact.mjs")));
const receipt = await deliverPortableArtifact({ inputPath, outputPath }, {
  build: input => buildPortableArtifact(input).replace(
    "</head>", "<style>html,body{max-width:100%;overflow-x:hidden}</style>\n</head>"),
});
process.stdout.write(JSON.stringify(receipt) + "\n");
if (!receipt.ok) process.exitCode = 1;
