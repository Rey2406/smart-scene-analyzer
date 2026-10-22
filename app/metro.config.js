// Metro bundler config — Expo SDK 58.
//
// The default Metro config does not know about `.tflite` or `.pte` files. It
// silently drops them from the bundle at BUILD time -- there is no error here.
// The failure shows up later, at app LOAD time, on a `require('./assets/models/...')`
// that points at a path that looks correct on disk, because the file that path
// names was never packaged. Registering both extensions below is what makes the
// artifact contract (CLAUDE.md, "The artifact contract"; docs/architecture.md, C11)
// actually reach the device.
//
//   'tflite' — required by react-native-fast-tflite (YOLO11 detection model)
//   'pte'    — required by react-native-executorch (Depth Anything V2 depth model)
const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

config.resolver.assetExts.push('tflite', 'pte');

module.exports = config;
