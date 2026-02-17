import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: root
    required property var backend
    readonly property int selectColumnWidth: 28
    readonly property int separatorWidth: 1
    readonly property int cellPaddingX: 8
    readonly property int minFileColumnWidth: 180
    readonly property int minStreamColumnWidth: 120
    property int fileColumnWidth: 320
    property int audioColumnWidth: 250
    property int subtitleColumnWidth: 250
    readonly property int tableContentWidth: selectColumnWidth + (separatorWidth * 3) + fileColumnWidth + audioColumnWidth + subtitleColumnWidth + 16
    visible: true
    title: "Sub Manager"
    x: backend.windowX
    y: backend.windowY
    width: backend.windowW
    height: backend.windowH

    function statusBg(level) {
        if (level === "ok") return "#dff6e7"
        if (level === "error") return "#fde4e4"
        return "#fff4d6"
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

    function clamp(value, minValue, maxValue) {
        return Math.max(minValue, Math.min(maxValue, value))
    }

    onClosing: function(close) {
        if (!backend.requestClose()) {
            close.accepted = false
            return
        }
        backend.saveWindowGeometry(x, y, width, height)
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 10

        RowLayout {
            Layout.fillWidth: true
            Button {
                text: "Add Videos"
                onClicked: backend.addVideos()
            }
            Item { Layout.fillWidth: true }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Label { text: "FFmpeg status:" }
            RowLayout {
                Layout.fillWidth: true
                spacing: 6

                Rectangle {
                    width: 10
                    height: 10
                    radius: 5
                    color: statusBorder(backend.ffmpegStatusLevel)
                    border.width: 1
                    border.color: statusFg(backend.ffmpegStatusLevel)
                }
                Label {
                    Layout.fillWidth: true
                    text: backend.ffmpegStatus
                    wrapMode: Text.Wrap
                }
            }
            Button {
                text: "Download FFmpeg"
                enabled: !backend.isDownloading
                onClicked: backend.downloadFfmpeg()
            }
            Button {
                text: "Select FFmpeg Directory"
                enabled: !backend.isDownloading
                onClicked: backend.selectFfmpegDirectory()
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: 8
            color: "#fafafa"
            border.width: 1
            border.color: "#d9d9d9"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 8
                spacing: 6

                Flickable {
                    id: tableScroll
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    contentWidth: root.tableContentWidth
                    contentHeight: height
                    flickableDirection: Flickable.HorizontalFlick
                    interactive: contentWidth > width

                    function fillFileColumnIfNeeded() {
                        var used = root.selectColumnWidth + (root.separatorWidth * 3) + root.fileColumnWidth + root.audioColumnWidth + root.subtitleColumnWidth + 16
                        if (width > used) {
                            root.fileColumnWidth += (width - used)
                        }
                    }

                    Component.onCompleted: fillFileColumnIfNeeded()
                    onWidthChanged: fillFileColumnIfNeeded()

                    WheelHandler {
                        target: null
                        acceptedModifiers: Qt.ShiftModifier
                        onWheel: function(event) {
                            var raw = 0
                            if (event.pixelDelta.y !== 0 || event.pixelDelta.x !== 0) {
                                raw = event.pixelDelta.y !== 0 ? event.pixelDelta.y : event.pixelDelta.x
                            } else {
                                raw = event.angleDelta.y !== 0 ? event.angleDelta.y / 2 : event.angleDelta.x / 2
                            }
                            var maxX = Math.max(0, tableScroll.contentWidth - tableScroll.width)
                            tableScroll.contentX = root.clamp(tableScroll.contentX - raw, 0, maxX)
                            event.accepted = true
                        }
                    }

                    ScrollBar.horizontal: ScrollBar {
                        policy: ScrollBar.AlwaysOn
                        visible: tableScroll.contentWidth > tableScroll.width
                    }

                    Rectangle {
                    id: headerRect
                    width: tableScroll.contentWidth
                    height: 42
                    radius: 6
                    color: "#f0f0f0"
                    border.width: 1
                    border.color: "#d9d9d9"

                    Item {
                        id: headerContent
                        anchors.fill: parent
                        anchors.margins: 8

                        Item {
                            x: 0
                            width: root.selectColumnWidth
                            height: parent.height

                            CheckBox {
                                anchors.centerIn: parent
                                checked: backend.allFilesChecked
                                onToggled: backend.setAllFilesChecked(checked)
                            }
                        }

                        Rectangle {
                            x: root.selectColumnWidth
                            width: root.separatorWidth
                            height: parent.height
                            color: "#d0d0d0"
                        }

                        Label {
                            x: root.selectColumnWidth + root.separatorWidth
                            width: root.fileColumnWidth
                            height: parent.height
                            text: "File Name"
                            font.bold: true
                            horizontalAlignment: Text.AlignLeft
                            verticalAlignment: Text.AlignVCenter
                            leftPadding: root.cellPaddingX
                            rightPadding: root.cellPaddingX
                        }

                        Rectangle {
                            x: root.selectColumnWidth + root.separatorWidth + root.fileColumnWidth
                            width: root.separatorWidth
                            height: parent.height
                            color: "#d0d0d0"
                        }

                        Label {
                            x: root.selectColumnWidth + (root.separatorWidth * 2) + root.fileColumnWidth
                            width: root.audioColumnWidth
                            height: parent.height
                            text: "Audio"
                            horizontalAlignment: Text.AlignLeft
                            verticalAlignment: Text.AlignVCenter
                            font.bold: true
                            leftPadding: root.cellPaddingX
                            rightPadding: root.cellPaddingX
                        }

                        Label {
                            x: root.selectColumnWidth + (root.separatorWidth * 3) + root.fileColumnWidth + root.audioColumnWidth
                            width: root.subtitleColumnWidth
                            height: parent.height
                            text: "Subtitle"
                            horizontalAlignment: Text.AlignLeft
                            verticalAlignment: Text.AlignVCenter
                            font.bold: true
                            leftPadding: root.cellPaddingX
                            rightPadding: root.cellPaddingX
                        }

                        Rectangle {
                            x: root.selectColumnWidth + (root.separatorWidth * 2) + root.fileColumnWidth + root.audioColumnWidth
                            width: root.separatorWidth
                            height: parent.height
                            color: "#d0d0d0"
                        }

                        Rectangle {
                            x: root.selectColumnWidth + root.separatorWidth + root.fileColumnWidth - 4
                            width: 8
                            height: parent.height
                            color: "transparent"
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.SplitHCursor
                                preventStealing: true
                                property real startSceneX: 0
                                property int startFileWidth: 0
                                onPressed: function(mouse) {
                                    tableScroll.interactive = false
                                    startSceneX = parent.mapToItem(headerRect, mouse.x, mouse.y).x
                                    startFileWidth = root.fileColumnWidth
                                }
                                onPositionChanged: function(mouse) {
                                    var currentX = parent.mapToItem(headerRect, mouse.x, mouse.y).x
                                    var delta = currentX - startSceneX
                                    root.fileColumnWidth = Math.max(root.minFileColumnWidth, startFileWidth + delta)
                                }
                                onReleased: tableScroll.interactive = tableScroll.contentWidth > tableScroll.width
                                onCanceled: tableScroll.interactive = tableScroll.contentWidth > tableScroll.width
                            }
                        }

                        Rectangle {
                            x: root.selectColumnWidth + (root.separatorWidth * 2) + root.fileColumnWidth + root.audioColumnWidth - 4
                            width: 8
                            height: parent.height
                            color: "transparent"
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.SplitHCursor
                                preventStealing: true
                                property real startSceneX: 0
                                property int startAudioWidth: 0
                                onPressed: function(mouse) {
                                    tableScroll.interactive = false
                                    startSceneX = parent.mapToItem(headerRect, mouse.x, mouse.y).x
                                    startAudioWidth = root.audioColumnWidth
                                }
                                onPositionChanged: function(mouse) {
                                    var currentX = parent.mapToItem(headerRect, mouse.x, mouse.y).x
                                    var delta = currentX - startSceneX
                                    root.audioColumnWidth = Math.max(root.minStreamColumnWidth, startAudioWidth + delta)
                                }
                                onReleased: tableScroll.interactive = tableScroll.contentWidth > tableScroll.width
                                onCanceled: tableScroll.interactive = tableScroll.contentWidth > tableScroll.width
                            }
                        }
                    }
                    }

                    Item {
                        y: headerRect.height + 6
                        width: tableScroll.contentWidth
                        height: tableScroll.height - headerRect.height - 6

                    ListView {
                        anchors.fill: parent
                        clip: true
                        spacing: 0
                        model: backend.videoFiles

                        delegate: Rectangle {
                            width: ListView.view.width
                            height: 42
                            radius: 6
                            border.width: 1
                            border.color: "#d9d9d9"
                            color: "white"

                            Item {
                                id: rowContent
                                anchors.fill: parent
                                anchors.margins: 8

                                Item {
                                    x: 0
                                    width: root.selectColumnWidth
                                    height: parent.height

                                    CheckBox {
                                        anchors.centerIn: parent
                                        checked: modelData.checked
                                        onToggled: backend.setFileChecked(index, checked)
                                    }
                                }

                                Rectangle {
                                    x: root.selectColumnWidth
                                    width: root.separatorWidth
                                    height: parent.height
                                    color: "#e0e0e0"
                                }

                                Label {
                                    x: root.selectColumnWidth + root.separatorWidth
                                    width: root.fileColumnWidth
                                    height: parent.height
                                    text: modelData.name
                                    elide: Label.ElideMiddle
                                    horizontalAlignment: Text.AlignLeft
                                    verticalAlignment: Text.AlignVCenter
                                    leftPadding: root.cellPaddingX
                                    rightPadding: root.cellPaddingX
                                }

                                Rectangle {
                                    x: root.selectColumnWidth + root.separatorWidth + root.fileColumnWidth
                                    width: root.separatorWidth
                                    height: parent.height
                                    color: "#e0e0e0"
                                }

                                Item {
                                    x: root.selectColumnWidth + (root.separatorWidth * 2) + root.fileColumnWidth
                                    width: root.audioColumnWidth
                                    height: parent.height
                                    clip: true

                                    Flickable {
                                        anchors.fill: parent
                                        anchors.leftMargin: root.cellPaddingX
                                        anchors.rightMargin: root.cellPaddingX
                                        contentWidth: audioChipRow.width
                                        contentHeight: height
                                        flickableDirection: Flickable.HorizontalFlick
                                        interactive: audioChipRow.width > width

                                        Row {
                                            id: audioChipRow
                                            spacing: 6
                                            anchors.verticalCenter: parent.verticalCenter

                                            Repeater {
                                                model: modelData.audio_language_items
                                                delegate: Rectangle {
                                                    radius: 10
                                                    height: 24
                                                    color: "#eef3ff"
                                                    border.width: 1
                                                    border.color: "#b5c7f5"
                                                    implicitWidth: chipText.implicitWidth + 14

                                                    Text {
                                                        id: chipText
                                                        anchors.centerIn: parent
                                                        text: modelData
                                                        font.pixelSize: 12
                                                        color: "#1f2d52"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                Rectangle {
                                    x: root.selectColumnWidth + (root.separatorWidth * 2) + root.fileColumnWidth + root.audioColumnWidth
                                    width: root.separatorWidth
                                    height: parent.height
                                    color: "#e0e0e0"
                                }
                                Item {
                                    x: root.selectColumnWidth + (root.separatorWidth * 3) + root.fileColumnWidth + root.audioColumnWidth
                                    width: root.subtitleColumnWidth
                                    height: parent.height
                                    clip: true

                                    Flickable {
                                        anchors.fill: parent
                                        anchors.leftMargin: root.cellPaddingX
                                        anchors.rightMargin: root.cellPaddingX
                                        contentWidth: subtitleChipRow.width
                                        contentHeight: height
                                        flickableDirection: Flickable.HorizontalFlick
                                        interactive: subtitleChipRow.width > width

                                        Row {
                                            id: subtitleChipRow
                                            spacing: 6
                                            anchors.verticalCenter: parent.verticalCenter

                                            Repeater {
                                                model: modelData.subtitle_language_items
                                                delegate: Rectangle {
                                                    radius: 10
                                                    height: 24
                                                    color: "#f0f8ee"
                                                    border.width: 1
                                                    border.color: "#b8d6ae"
                                                    implicitWidth: subtitleChipText.implicitWidth + 14

                                                    Text {
                                                        id: subtitleChipText
                                                        anchors.centerIn: parent
                                                        text: modelData
                                                        font.pixelSize: 12
                                                        color: "#244a1f"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Label {
                        anchors.centerIn: parent
                        visible: backend.videoFiles.length === 0
                        text: "No video files added"
                        color: "#666666"
                    }
                }
                }
            }
        }
    }
}
