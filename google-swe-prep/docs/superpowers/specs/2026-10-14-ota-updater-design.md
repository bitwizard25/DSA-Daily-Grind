# Proposed OTA updates for Google SWE II OS

Status: approved by the user on 14 October 2026; updater code has not been implemented.

## What the app will do

The installed Windows app will check for updates in the background while it is running. When internet access becomes available, it will download an approved update from this repository, verify the download, and install it after all app windows are closed. The next launch will use the updated version. A study session will never be interrupted by an automatic restart.

Automatic updates will be enabled by default, with an off switch and a **Check now** button in Settings. Existing installations need one manual upgrade to receive this feature. No background service or Windows scheduled task is proposed: checks run while the app is open, and a downloaded update can finish installing after it closes.

## Delivery choice

Use GitHub Releases with the existing Inno Setup installer. This fits the current public repository and per-user Windows installation. GitHub's public release API does not require a user's GitHub token.

Direct Actions artifact downloads would couple users to artifact retention and authenticated downloads. Adopting a different updater framework would require replacing more of the existing packaging. Neither is selected for this first implementation.

## CI publishes the update

1. Keep the current Windows, Linux and macOS test/build jobs.
2. Publish updates only from successful `main` builds, after every matrix job and the Windows installer smoke test succeeds. Development branch builds cannot become automatic updates.
3. Embed the app version, source commit and workflow build number in build metadata. Compare both app version and build number so a newly tested build is detected even when the source version string has not changed.
4. Publish a separate immutable release for each build, using a `google-prep-` tag prefix. Include the Windows installer and a small update manifest containing the application ID, platform, architecture, version, build number, commit, installer size and SHA-256 checksum.
5. Serialize publishing and prevent an older build completing late from replacing a newer stable build. Re-running a published build must not overwrite its assets.

## The app checks and downloads

- Run network work in a Python background worker, independently of browser timers and heartbeats. The interface and local study features remain available offline.
- Check at launch and at most every six hours after a successful check. After a connection failure, retry with increasing delays: one minute, five minutes, fifteen minutes, then hourly. Respect GitHub rate-limit responses and avoid overlapping checks/downloads.
- Consider only this repository's stable `google-prep-` releases with a valid manifest. Reject another product, unsupported platform, malformed version, incomplete release or older build.
- Use HTTPS and restricted GitHub download destinations. The installed app contains no GitHub credentials. Never accept an arbitrary installer URL supplied by the web interface.
- Download to a partial file outside the installation directory. Validate size and SHA-256 before promoting it to a pending update. A failed or interrupted download stays uninstalled and can be retried.
- Settings shows the current version/build, last check, available update, download state and pending install. Offline status is quiet; a manual check can explain a failure.

## Install when it is safe

- Automatic installation supports the registered Windows x64 Inno Setup installation. Source runs, portable executables and other operating systems receive manual download guidance rather than having their files replaced automatically.
- Stage a small standalone update helper outside the app directory so it can continue after the main EXE exits and avoid locking files the installer needs to replace.
- The helper waits for the launching process and all app instances to close, using the existing application mutex. It does not kill the app or unrelated browser processes. If files are still in use, defer installation instead of forcing other programs closed.
- Recheck the pending installer's checksum and the registered installation directory immediately before running it.
- Run the existing installer silently, with no forced system restart, no forced application closure and a bounded timeout. Preserve the installer AppId, installation location and existing preferences for shortcuts/PATH.
- Keep progress, design notes, STAR stories, lesson history and saved sessions in `.google_prep`, outside the replaced application files. Additive migrations must preserve older data; back up the database before the first launch of an updated build.
- Record the install result. Clear the pending update only after success and confirm the new build on the next launch. Do not repeatedly install the same build. Keep failures visible in Settings and retain a manual repair path; do not promise transactional rollback from the current installer.

## Verification required

- Version/build comparison, release filtering, manifest validation and downgrade rejection.
- Offline operation, connection recovery, rate-limit handling and overlapping-check prevention.
- Interrupted downloads, checksum/size mismatch and rejection of unsupported URLs.
- Deferral during active work, multiple running instances and locked files.
- Installer success/failure/timeout and preservation of shortcut/PATH preferences.
- Windows CI upgrade test from an older installer to the new one, including seeded progress, saved notes and session recovery; verify the updated app starts and the update is not applied twice.
- Source/portable mode performs no automatic installation.

## Sources checked

- Existing `gui/launcher.py`: dedicated browser process, launcher-held application mutex and shutdown lifecycle.
- Existing `installer/google-prep.iss`: per-user installation, stable AppId, mutex and retained user data.
- Existing `build-exe.yml`: matrix checks, installer smoke test and Actions artifact upload; no release-publishing job currently exists.
- [GitHub release API](https://docs.github.com/en/rest/releases/releases): public release lookup and asset metadata.
- [GitHub Actions artifacts API](https://docs.github.com/en/rest/actions/artifacts): authenticated artifact-download flow.
- [Inno Setup command-line parameters](https://jrsoftware.org/ishelp/topic_setupcmdline.htm): silent installation and restart/application-close controls.
