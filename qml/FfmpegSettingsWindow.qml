import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Window {
    id: root
    property var backend: null
    visible: false
    modality: Qt.NonModal
    transientParent: null
    width: 700
    height: 180
    minimumWidth: 560
    minimumHeight: 160
    title: "FFmpeg Settings"
    flags: Qt.Window
        | Qt.WindowTitleHint
        | Qt.WindowSystemMenuHint
        | Qt.WindowMinimizeButtonHint
        | Qt.WindowMaximizeButtonHint
        | Qt.WindowCloseButtonHint

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

    Rectangle {
        anchors.fill: parent
        color: "#ffffff"

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 14
            spacing: 10

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Label { text: "Status:" }
                Rectangle {
                    width: 10
                    height: 10
                    radius: 5
                    color: statusBorder(root.backend ? root.backend.ffmpegStatusLevel : "warn")
                    border.width: 1
                    border.color: statusFg(root.backend ? root.backend.ffmpegStatusLevel : "warn")
                }
                Label {
                    Layout.fillWidth: true
                    text: root.backend ? root.backend.ffmpegStatus : "Loading FFmpeg status..."
                    wrapMode: Text.Wrap
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Button {
                    text: "Download FFmpeg"
                    enabled: root.backend && !root.backend.isDownloading
                    onClicked: {
                        if (root.backend) {
                            root.backend.downloadFfmpeg()
                        }
                    }
                }
                Button {
                    text: "Select FFmpeg Directory"
                    enabled: root.backend && !root.backend.isDownloading
                    onClicked: {
                        if (root.backend) {
                            root.backend.selectFfmpegDirectory()
                        }
                    }
                }
                Item { Layout.fillWidth: true }
            }
        }
    }
}
