#define AppName "Sub Manager"
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef BuildOutputDir
  #define BuildOutputDir AddBackslash(SourcePath) + "..\\..\\dist\\sub-manager"
#endif
#ifndef OutputDir
  #define OutputDir AddBackslash(SourcePath) + "..\\..\\dist"
#endif

[Setup]
AppId={{A8B56C19-A67D-4D03-9F6C-5F9867EFA12A}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=jollywitch
DefaultDirName={autopf}\Sub Manager
DefaultGroupName=Sub Manager
UninstallDisplayIcon={app}\sub-manager.exe
OutputDir={#OutputDir}
OutputBaseFilename=sub-manager-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "{#BuildOutputDir}\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\\Sub Manager"; Filename: "{app}\\sub-manager.exe"
Name: "{group}\\Uninstall Sub Manager"; Filename: "{uninstallexe}"
Name: "{autodesktop}\\Sub Manager"; Filename: "{app}\\sub-manager.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\\sub-manager.exe"; Description: "Launch Sub Manager"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  RemoveAllData: Boolean;
  AppDataDir: string;
  GlmCacheDir: string;
  UserProfileDir: string;
begin
  if CurUninstallStep <> usUninstall then
    exit;

  RemoveAllData :=
    MsgBox(
      'Do you also want to remove all Sub Manager user data?' + #13#10 + #13#10 +
      'This will delete:' + #13#10 +
      '- App data in %LOCALAPPDATA%\sub-manager (logs, downloaded tools, runtime packages)' + #13#10 +
      '- Downloaded GLM model cache in %USERPROFILE%\.cache\huggingface\hub\models--zai-org--GLM-OCR' + #13#10 +
      '- Saved app settings (including HF token)',
      mbConfirmation,
      MB_YESNO
    ) = IDYES;

  if not RemoveAllData then
    exit;

  AppDataDir := AddBackslash(GetEnv('LOCALAPPDATA')) + 'sub-manager';
  UserProfileDir := GetEnv('USERPROFILE');
  GlmCacheDir := AddBackslash(UserProfileDir) + '.cache\huggingface\hub\models--zai-org--GLM-OCR';

  if DirExists(AppDataDir) then
    DelTree(AppDataDir, True, True, True);
  if DirExists(GlmCacheDir) then
    DelTree(GlmCacheDir, True, True, True);

  RegDeleteKeyIncludingSubkeys(HKCU, 'Software\sub-manager\sub-manager');
  RegDeleteKeyIncludingSubkeys(HKCU, 'Software\sub-manager');
end;
