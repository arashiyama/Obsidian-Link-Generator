import { readFileSync, writeFileSync } from "fs";

const targetVersion = process.argv[2];
if (!targetVersion) {
  console.error("Please specify a target version");
  process.exit(1);
}

// Read manifest.json
const manifestPath = "manifest.json";
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));

// Update version in manifest.json
const currentVersion = manifest.version;
manifest.version = targetVersion;
writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

// Read versions.json
const versionsPath = "versions.json";
const versions = JSON.parse(readFileSync(versionsPath, "utf8"));

// Add/update the version
versions[targetVersion] = manifest.minAppVersion;
writeFileSync(versionsPath, JSON.stringify(versions, null, 2));

console.log(`Updated version from ${currentVersion} to ${targetVersion}`);
