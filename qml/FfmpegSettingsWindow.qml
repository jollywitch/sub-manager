import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Window {
    id: root
    property var backend: null
    property string hfTokenInput: ""
    readonly property int dependencyRowHeight: 32
    readonly property color detailTextColor: "#777777"
    visible: false
    modality: Qt.ApplicationModal
    width: dependenciesLayout.implicitWidth + 24
    height: dependenciesLayout.implicitHeight + 24
    minimumWidth: dependenciesLayout.implicitWidth + 24
    minimumHeight: dependenciesLayout.implicitHeight + 24
    title: "Dependencies"
    flags: Qt.Dialog
        | Qt.WindowTitleHint
        | Qt.WindowSystemMenuHint
        | Qt.WindowCloseButtonHint

    onVisibleChanged: {
        if (visible && root.backend) {
            root.backend.refreshDependencyStatuses()
        }
    }

    function statusFg(level) {
        if (level === "ok") return "#0a5a2a"
        if (level === "error") return "#7a0f0f"
        return "#5c3d00"
    }

    function statusBorder(level) {
        if (level === "ok") return "#91d0a8"
        if (level === "error") return "#ea9a9a"
        return "#efc76f"
    }

    function indicatorLabel(level) {
        if (level === "ok") return "Ready"
        if (level === "progress") return "In Progress"
        if (level === "error") return "Error"
        return "Not Ready"
    }

    function statusDetail(rawText, preferredDetail) {
        const preferred = (preferredDetail || "").trim()
        if (preferred.length > 0) {
            return preferred
        }
        const text = (rawText || "").trim()
        const colonIndex = text.indexOf(":")
        if (colonIndex >= 0) {
            return text.slice(colonIndex + 1).trim()
        }
        const match = text.match(/^(.*)\(([^)]+)\)\.?$/)
        if (match) {
            return match[2].trim()
        }
        return text
    }

    Rectangle {
        anchors.fill: parent
        color: "#ffffff"

        ColumnLayout {
            id: dependenciesLayout
            anchors.fill: parent
            anchors.margins: 12
            spacing: 2

            RowLayout {
                Layout.fillWidth: true
                Layout.minimumHeight: root.dependencyRowHeight
                Layout.preferredHeight: root.dependencyRowHeight
                spacing: 6

                Label {
                    text: "FFmpeg/FFprobe:"
                    Layout.preferredWidth: 160
                }
                Rectangle {
                    width: 10
                    height: 10
                    radius: 5
                    color: statusBorder(root.backend ? root.backend.ffmpegStatusLevel : "warn")
                    border.width: 1
                    border.color: statusFg(root.backend ? root.backend.ffmpegStatusLevel : "warn")
                }
                Label {
                    text: root.backend ? root.indicatorLabel(root.backend.ffmpegStatusLevel) + ":" : "Loading:"
                    wrapMode: Text.NoWrap
                    elide: Text.ElideRight
                }
                Label {
                    Layout.fillWidth: true
                    text: root.statusDetail(root.backend ? root.backend.ffmpegStatus : "Loading FFmpeg status...", "")
                    color: root.detailTextColor
                    wrapMode: Text.NoWrap
                    elide: Text.ElideMiddle
                }

                Button {
                    text: "Download"
                    enabled: root.backend && !root.backend.isDownloading
                    onClicked: {
                        if (root.backend) {
                            root.backend.downloadFfmpeg()
                        }
                    }
                }
                Button {
                    text: "Select Directory"
                    enabled: root.backend && !root.backend.isDownloading
                    onClicked: {
                        if (root.backend) {
                            root.backend.selectFfmpegDirectory()
                        }
                    }
                }
                Button {
                    text: "Search PATH"
                    enabled: root.backend && !root.backend.isDownloading
                    onClicked: {
                        if (root.backend) {
                            root.backend.searchFfmpegInPath()
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.minimumHeight: root.dependencyRowHeight
                Layout.preferredHeight: root.dependencyRowHeight
                spacing: 6

                Label {
                    text: "HF Token (optional):"
                    Layout.preferredWidth: 160
                }
                Rectangle {
                    width: 10
                    height: 10
                    radius: 5
                    color: statusBorder(root.backend ? root.backend.hfTokenStatusLevel : "warn")
                    border.width: 1
                    border.color: statusFg(root.backend ? root.backend.hfTokenStatusLevel : "warn")
                }
                Label {
                    text: root.backend ? root.indicatorLabel(root.backend.hfTokenStatusLevel) + ":" : "Loading:"
                    wrapMode: Text.NoWrap
                    elide: Text.ElideRight
                }
                Label {
                    Layout.preferredWidth: 190
                    text: root.backend && root.backend.hfTokenPreview.length > 0
                        ? root.backend.hfTokenPreview
                        : ""
                    color: root.detailTextColor
                    font.family: "monospace"
                    wrapMode: Text.NoWrap
                    elide: Text.ElideMiddle
                }
                TextField {
                    id: hfTokenField
                    Layout.fillWidth: true
                    echoMode: TextInput.Password
                    placeholderText: "Enter HF token"
                    text: root.hfTokenInput
                    onTextChanged: root.hfTokenInput = text
                }
                Button {
                    text: "Save"
                    enabled: root.backend && hfTokenField.text.trim().length > 0
                    onClicked: {
                        root.backend.setHfToken(hfTokenField.text)
                        hfTokenField.text = ""
                    }
                }
                Button {
                    text: "Clear"
                    enabled: root.backend
                    onClicked: {
                        root.backend.clearHfToken()
                        hfTokenField.text = ""
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.minimumHeight: root.dependencyRowHeight
                Layout.preferredHeight: root.dependencyRowHeight
                spacing: 6

                Label {
                    text: "GLM-OCR Model:"
                    Layout.preferredWidth: 160
                }
                Rectangle {
                    width: 10
                    height: 10
                    radius: 5
                    color: statusBorder(root.backend ? root.backend.glmOcrModelStatusLevel : "warn")
                    border.width: 1
                    border.color: statusFg(root.backend ? root.backend.glmOcrModelStatusLevel : "warn")
                }
                Label {
                    text: root.backend ? root.indicatorLabel(root.backend.glmOcrModelStatusLevel) + ":" : "Loading:"
                    wrapMode: Text.NoWrap
                    elide: Text.ElideRight
                }
                Label {
                    Layout.fillWidth: true
                    text: root.statusDetail(root.backend ? root.backend.glmOcrModelStatus : "Loading GLM-OCR model status...", "")
                    color: root.detailTextColor
                    wrapMode: Text.NoWrap
                    elide: Text.ElideMiddle
                }
                Item { Layout.preferredWidth: 250 }
            }
        }
    }
}
