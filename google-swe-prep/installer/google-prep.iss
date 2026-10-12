; Windows installer for Google SWE II OS (Inno Setup 6).
; Built by `python build.py --installer`, which first builds dist\GoogleSWEPrep\ from installer\app.spec
; and passes the defines below. Installs per user (no admin prompt) into %LOCALAPPDATA%\Programs,
; adds Start menu and desktop shortcuts and an entry in Settings > Apps for uninstalling.
; Your progress lives in %USERPROFILE%\.google_prep and is never touched by install, update or uninstall.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\GoogleSWEPrep"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif
#ifndef IconFile
  #define IconFile "..\build\google-prep.ico"
#endif
#define AppName "Google SWE II OS"
#define AppExe "google-prep.exe"

[Setup]
; The AppId ties every future version to this install: a newer Setup updates it in place.
AppId={{03EDDFF2-4DC8-4E0E-84DE-FA73BFFCC5D8}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=DSA-Daily-Grind
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableReadyPage=yes
PrivilegesRequired=lowest
OutputDir={#OutputDir}
OutputBaseFilename=GoogleSWEPrep-Setup
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; the app holds this mutex while open (launcher.APP_MUTEX): updating or uninstalling asks you to close it first
AppMutex=GoogleSWEIIOS.Running
CloseApplications=yes
RestartApplications=no
ChangesEnvironment=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "addtopath"; Description: "Let terminals run google-prep-cli (adds the app folder to your PATH)"; GroupDescription: "Terminal:"; Flags: unchecked

[InstallDelete]
; drop the previous version's libraries so an update never mixes old and new files
; (only when updating this app: never in a folder that holds some other program)
Type: filesandordirs; Name: "{app}\_internal"; Check: IsOurUpdate

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; Comment: "Daily Google interview prep"
Name: "{autoprograms}\{#AppName} (terminal)"; Filename: "{app}\google-prep-cli.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
  Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
function IsOurUpdate: Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\{#AppExe}')) and FileExists(ExpandConstant('{app}\google-prep-cli.exe'));
end;

function NeedsAddPath(Dir: string): Boolean;
var
  Paths: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Paths) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') = 0;
end;

procedure RemoveFromPath(Dir: string);
var
  Paths: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Paths) then
    exit;
  P := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';');
  if P = 0 then
    exit;
  if P = 1 then
    Delete(Paths, 1, Length(Dir) + 1)  { first entry: drop "Dir;" }
  else
    Delete(Paths, P - 1, Length(Dir) + 1);  { later entry: drop ";Dir" }
  RegWriteStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Paths);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    RemoveFromPath(ExpandConstant('{app}'));
end;
